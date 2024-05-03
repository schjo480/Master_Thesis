# Implementation of the Gretel baseline
# https://arxiv.org/pdf/1903.07518.pdf
# Reference implementation: https://github.com/jbcdnr/gretel-path-extrapolation/blob/master/main.py

from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from omegaconf import DictConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


from trajectory_gnn.model.layers import * # import all such that hydra can instantiate
from trajectory_gnn.model.base import TrajectoryForecasting
from trajectory_gnn.data.base import BaseDataset, TrajectoryComplex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingDataset, TrajectoryForecastingPrediction
from trajectory_gnn.utils.graph import add_self_loops_to_degree_zero_nodes, graph_diffusion
from trajectory_gnn.config import GretelConfig

from hydra.utils import instantiate

from trajectory_gnn.utils.utils import scatter_arange, scatter_normalize_positive

class GretelMLP(nn.Module):
    """ MLP in gretel composed of two layers according to: https://github.com/jbcdnr/gretel-path-extrapolation/blob/master/model.py"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 2 * in_dim # don't ask me why, that's what Gretel does, see reference code
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x: TensorType['n', 'in_dim']) -> TensorType['n', 'out_dim']:
            
        return self.lin2(torch.sigmoid(self.lin1(x)))
    

class Gretel(TrajectoryForecasting):
    
    def __init__(self, config: GretelConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)

        self.diffusion_edge_mlp = GretelMLP(
            2 * dataset.base_dataset.num_node_features + dataset.base_dataset.num_edge_features,
            1)
        
        # Gretel can only operate on fixed length input sequences as it processes with an MLP
        self.num_observations = config.num_observations 
        self.parametric_diffusion = config.parametric_diffusion
        self.num_diffusion_steps = config.num_diffusion_steps
        self.left_padding = config.left_padding
        if self.parametric_diffusion:
            self.diffusion_hidden_dim = config.diffusion_hidden_dim
            self.diffusion_lift = nn.Linear(1, self.diffusion_hidden_dim)
            self.diffusion_linears = nn.ModuleList([nn.Linear(
                self.diffusion_hidden_dim,
                self.diffusion_hidden_dim) for _ in range(config.num_diffusion_steps)])
        self.non_backtracking_random_walk = config.non_backtracking_random_walk
        
        self.output_mlp = GretelMLP(self.pseudo_coordinate_dim * 2 * self.num_observations +
                                     dataset.base_dataset.num_node_features * 2 +
                                     dataset.base_dataset.num_edge_features , 1)
    
    @property
    def name(self) -> str:
        return 'gretel'
       
    @property
    def pseudo_coordinate_dim(self) -> int:
        if self.parametric_diffusion: 
            return self.diffusion_hidden_dim
        else:
            return 1
        
    @property
    def multiple_steps_predictions_differientable(self) -> bool:
        return True  

    @property
    def receptive_field(self) -> int | None:
        return self.num_diffusion_steps

    @typechecked
    def _parametric_diffusion(self, x: TensorType['num_nodes', 'num_observations', 1], diffusion_edges: TensorType[2, 'num_edges'], 
                   diffusion_edge_weights: TensorType['num_edges', 1]) -> TensorType['num_nodes', 'num_observations', 'out_dim_diffusion']:
        idx_src, idx_target = diffusion_edges
        x = self.diffusion_lift(x)
        diffusion_edge_weights = diffusion_edge_weights.view(-1, 1, 1) # every obsevation has an independent diffusion process, and they're all the same
        for step in range(self.num_diffusion_steps):
            message = x[idx_src] * diffusion_edge_weights
            x = torch_scatter.scatter_add(message, idx_target, dim=0, dim_size=x.size(0))
            x = self.diffusion_linears[step](x)
            x = F.relu(x, inplace=True)
        return x
    
    @typechecked
    def _nonparametric_diffusion(self, x: TensorType['num_nodes', 'num_observations', 1], diffusion_edges: TensorType[2, 'num_edges'], 
                   diffusion_edge_weights: TensorType['num_edges', 1]) -> TensorType['num_nodes', 'num_observations', 1]:
        return graph_diffusion(x, diffusion_edges, diffusion_edge_weights.view(-1, 1, 1), self.num_diffusion_steps, return_all_steps=False)
           
    @typechecked
    def _compute_initial_node_probabilities(self, x: List[TensorType], batch: TrajectoryComplex) -> TensorType['num_nodes', 'num_observations', 1]:
        """ Computes the initial observation probabilities to be diffused. """
        x_nodes = x[0]
        node_idxs, trajectory_idxs = batch.trajectory_node_idxs
        node_probabilites = torch.zeros(x_nodes.size(0), self.num_observations, device=x_nodes.device)
        
        # We also handle observing more than `self.num_observations` nodes by cutting the trajectory and only using the suffix
        # We handle observing less than `self.num_observations` by "left padding" nodes, i.e. assigning zero probability on timesteps 0, ... t
        # for which we have no observations
        trajectory_sizes = torch_scatter.scatter_add(torch.ones_like(trajectory_idxs), trajectory_idxs)
        timesteps = scatter_arange(trajectory_idxs)
        mask_cutoff = timesteps >= (trajectory_sizes - self.num_observations)[trajectory_idxs]
        # Apply the cutoff
        node_idxs, trajectory_idxs = node_idxs[mask_cutoff], trajectory_idxs[mask_cutoff]
        timesteps = scatter_arange(trajectory_idxs)
        trajectory_sizes = torch_scatter.scatter_add(torch.ones_like(trajectory_idxs), trajectory_idxs)
        # Pad short trajectories by addind to the timestep index
        if self.left_padding:
            padding = torch.clip(self.num_observations - trajectory_sizes, min=0)
            timesteps += padding[trajectory_idxs]

        # For the datasets mapped to graphs: 1-hot vectors (all probability mass on one node, i.e. the observed node)
        node_probabilites[node_idxs, timesteps] = 1.0
        node_probabilites = node_probabilites.unsqueeze(-1) # num_nodes, num_observations, 1
        return node_probabilites
        
    @typechecked
    def compute_random_walk_matrix_weights(self, x: List[TensorType], batch: TrajectoryComplex,
                                           ) -> TensorType['num_node_adj', 1]:
        """ Computes the weights of the random walk matrix on nodes"""
        x_nodes, x_edges = x[0], x[1]
        
        idx_src, idx_target = batch.edge_complex.node_adjacency
        diffusion_edge_weights = self.diffusion_edge_mlp(torch.cat([
            x_nodes[idx_src], x_nodes[idx_target], x_edges
        ], dim=-1))
        assert diffusion_edge_weights.size(-1) == 1
        diffusion_edge_weights = torch_scatter.scatter_softmax(diffusion_edge_weights,
                                                               idx_src, dim=0)
        
        node_pseudo_coordinates = self._compute_initial_node_probabilities(x, batch)
        if self.parametric_diffusion:
            node_pseudo_coordinates = self._parametric_diffusion(node_pseudo_coordinates, batch.edge_complex.node_adjacency,
                                                 diffusion_edge_weights)
        else:
            node_pseudo_coordinates = self._nonparametric_diffusion(node_pseudo_coordinates, batch.edge_complex.node_adjacency,
                                                    diffusion_edge_weights)
        # h_nodes: num_nodes, num_observations, pseudo_coordinate_dim
        # concatenate all observations (i.e. timesteps)
        node_pseudo_coordinates = node_pseudo_coordinates.view(x_nodes.size(0), -1)
        output = self.output_mlp(torch.cat([
            node_pseudo_coordinates[idx_src], node_pseudo_coordinates[idx_target],
            x_nodes[idx_src], x_nodes[idx_target], x_edges[batch.edge_complex.node_adjacency_edge_idxs]
            ], dim=-1)) # num_edges

        # normalize to have a proper random walk
        output = torch_scatter.scatter_softmax(output, batch.edge_complex.node_adjacency[0],
                                                        dim=0)
        return output
    
    @typechecked
    def prepare_inputs(self, batch: TrajectoryComplex) -> Tuple[List[TensorType | None], TrajectoryComplex]:
        assert all(g.directed for g in batch.edge_complex.graphs), f'Gretel models edges as directed, so the input graphs should be directed as well.'
        # Gretel to some degree also can work with undirected edges, but 
        # i) how do you initialize the edge distribution? you already lose directionality information there, 
        # which is important for the non backtracking random walk
        # ii) how do you map from edge distributions to a node distribution? here, also directionality is needed
        return super().prepare_inputs(batch)
    
    @typechecked
    def forward(self, batch: TrajectoryComplex, context: str) -> TrajectoryForecastingPrediction:
        return self.predict_multiple_steps(batch, range(1), context)[0]
        
    @typechecked
    def _compute_edge_adjacency(self, x: List[TensorType], batch: TrajectoryComplex) -> Tuple[TensorType[2, 'num_edge_adj'], TensorType['num_edge_adj', 1],
                                                                                              TensorType['num_node_adj']]:
        """ Computes the random walk matrix on edges. """
        node_random_walk_matrix_weights = self.compute_random_walk_matrix_weights(x, batch) # num_node_adj, 1
        edge_adjacency_non_backtracking, edge_random_walk_matrix_weights = self._node_random_walk_matrix_to_edge_random_walk_matrix(batch, node_random_walk_matrix_weights)
        return edge_adjacency_non_backtracking, edge_random_walk_matrix_weights, node_random_walk_matrix_weights.view(-1)
    
    @typechecked
    def _node_random_walk_matrix_to_edge_random_walk_matrix(self, batch: TrajectoryComplex, 
                                                            node_random_walk_matrix_weights: TensorType['num_node_adj', 1]) -> Tuple[
                                                                TensorType[2, 'num_edge_adj_non_backtracking'],
                                                                TensorType['num_edge_adj_non_backtracking', 1]]:
        """ Transforms a positive node random walk matrix to a non-backtracking random walk on the edges of the graph. """
        if self.non_backtracking_random_walk:
            non_backtracking_idxs = batch.edge_complex.non_backtracking_random_walk_edge_adjacency_idxs
            edge_adjacency_non_backtracking = batch.edge_complex.edge_adjacency[:, non_backtracking_idxs] # 2, num_non_backtracking
            edge_random_walk_matrix_weights = node_random_walk_matrix_weights[batch.edge_complex.edge_adjacency_idx_to_node_adjacency_idx[non_backtracking_idxs]]
        else:
            edge_adjacency_non_backtracking = batch.edge_complex.edge_adjacency
            edge_random_walk_matrix_weights = node_random_walk_matrix_weights[batch.edge_complex.edge_adjacency_idx_to_node_adjacency_idx]
        
        edge_adjacency_non_backtracking, edge_random_walk_matrix_weights = add_self_loops_to_degree_zero_nodes(edge_adjacency_non_backtracking, num_nodes=batch.edge_complex.num_edges,
                                                                                                               fill_value=1.0, weights=edge_random_walk_matrix_weights)
        edge_random_walk_matrix_weights = scatter_normalize_positive(edge_random_walk_matrix_weights, edge_adjacency_non_backtracking[0])
        return edge_adjacency_non_backtracking, edge_random_walk_matrix_weights
    
    @typechecked
    def predict_multiple_steps_diffusion(self, batch: TrajectoryComplex, steps: Iterable[int], context: str) -> List[TrajectoryForecastingPrediction | None]:
        """ Predicts for multiple time steps in the future using graph diffusion over the learned Markov Chain.
        Note that this does not actually generate a path, just probability distribution over nodes and edges """
        x, batch = self.prepare_inputs(batch)
        steps = set(steps)
        max_num_steps = max(steps) + 1
        
        edge_adjacency_non_backtracking, edge_random_walk_matrix_weights, node_rw_weights = self._compute_edge_adjacency(x, batch)
        assert node_rw_weights.size(0) == batch.edge_complex.num_edges
        p_edge = torch.zeros(batch.edge_complex.num_edges, device=edge_random_walk_matrix_weights.device)
        endpoint_edge_idxs, _, _ = batch.trajectory_endpoints
        p_edge[endpoint_edge_idxs] = 1.0
        
        # Note that the most likely node in k steps does not necessarily correspond to the node obtained by
        # greedily sampling for k steps from the markov transition matrix
        # therefore, for the diffusion process, we can't actually greedily sample a path that matches it
        # one would have to use DP to compute the most likely path
        # here, we sample a path greedily, but the random walk probabilities (on which the model is trained)
        # come from the diffusion process. Therefore, the last sampled step will not be the node with maximal
        # score overall
        diffused = graph_diffusion(p_edge.view(-1, 1), edge_adjacency_non_backtracking, edge_random_walk_matrix_weights,
                                num_steps=max_num_steps, return_all_steps=True)
        predictions = []
        for step_idx, edge_distribution in enumerate(diffused):
            if step_idx in steps:
                prediction = self._edge_distribution_to_prediction(batch, edge_distribution)
            else:
                with torch.no_grad():
                    prediction = self._edge_distribution_to_prediction(batch, edge_distribution)
            
            prediction.populate_successor_candidates_from_batch(batch)
            prediction.sampled_successor_candidate_idxs = self.sample_successors(batch, prediction)
            prediction.node_random_walk_weights = node_rw_weights
            if step_idx < max_num_steps - 1:
                batch = batch.advance(
                    prediction.successor_candidate_edge_idxs[prediction.sampled_successor_candidate_idxs],
                    prediction.successor_candidate_edge_orientations[prediction.sampled_successor_candidate_idxs],
                    prediction.successor_candidate_trajectory_idxs[prediction.sampled_successor_candidate_idxs],
                )        
            predictions.append(prediction if step_idx in steps else None)
        return predictions

    @typechecked
    def predict_multiple_steps_greedy(self, batch: TrajectoryComplex, steps: Iterable[int], context: str) -> List[TrajectoryForecastingPrediction | None]:
        """ Predicts for multiple time step by greedily sampling a neighbour from the learned Markov Chain. """
        x, batch = self.prepare_inputs(batch)
        edge_adjacency_non_backtracking, edge_random_walk_matrix_weights, node_rw_weights = self._compute_edge_adjacency(x, batch)
        assert node_rw_weights.size(0) == batch.edge_complex.num_edges
        
        result = []
        steps = set(steps)
        max_num_steps = max(steps) + 1
        for step_idx in range(max_num_steps):
            p_edge = torch.zeros(batch.edge_complex.num_edges, device=edge_random_walk_matrix_weights.device)
            endpoint_edge_idxs, _, _ = batch.trajectory_endpoints
            p_edge[endpoint_edge_idxs] = 1.0
            p_edge = graph_diffusion(p_edge.view(-1, 1), edge_adjacency_non_backtracking, edge_random_walk_matrix_weights,
                                        num_steps=1, return_all_steps=True)[0]
            prediction = self._edge_distribution_to_prediction(batch, p_edge)
            with torch.no_grad():
                prediction.populate_successor_candidates_from_batch(batch)
                prediction.sampled_successor_candidate_idxs = self.sample_successors(batch, prediction)
                prediction.node_random_walk_weights = node_rw_weights
                if step_idx < max_num_steps - 1:
                    batch = batch.advance(
                        prediction.successor_candidate_edge_idxs[prediction.sampled_successor_candidate_idxs],
                        prediction.successor_candidate_edge_orientations[prediction.sampled_successor_candidate_idxs],
                        prediction.successor_candidate_trajectory_idxs[prediction.sampled_successor_candidate_idxs],
                    )
            result.append(prediction if step_idx in steps else None)
        return result
    
    @typechecked
    def predict_multiple_steps(self, batch: TrajectoryComplex, steps: Iterable[int], context: str) -> List[TrajectoryForecastingPrediction | None]:
        """ Predicts the node distribution for a given horizon. """
        if context == 'train':
            return self.predict_multiple_steps_diffusion(batch, steps, context)
        else:
            return self.predict_multiple_steps_greedy(batch, steps, context)
    
    @typechecked
    def sample_successors(self, batch: TrajectoryComplex, pred: TrajectoryForecastingPrediction) -> TensorType['num_trajectories']:
        """ Samples the successor indices from a data sample and the model predictions on it
        """
        # Gretel samples based on the edge transition probability
        return pred.get_sampled_successor_candidate_idxs_by_max_score(by_order=1)
        
    @typechecked
    def _edge_distribution_to_prediction(self, batch: TrajectoryComplex, edge_distribution: TensorType['num_edges', 1]) -> TrajectoryForecastingPrediction:
        """ Transforms a probability vector on edges into a prediction. """
        assert batch.edge_complex.node_adjacency.size(1) == edge_distribution.size(0), 'In a directed graph, the number of edges should match the size of the node adjacency'
        node_adjacency_idxs = batch.edge_complex.edge_idxs_to_node_adjacency_idxs(torch.arange(edge_distribution.size(0), device=edge_distribution.device))
        idx_src, idx_target = batch.edge_complex.node_adjacency[:, node_adjacency_idxs]
        node_distribution = torch_scatter.scatter_sum(edge_distribution, idx_target, dim=0, dim_size=batch.edge_complex.num_nodes)
        assert len(node_distribution.size()) == 2
        assert len(edge_distribution.size()) == 2
        return TrajectoryForecastingPrediction(probabilities=[node_distribution.view(-1), edge_distribution.view(-1)])
        
        
        
        
        
        
    
    