import torch
import torch.nn as nn
import torch_scatter
import numpy as np
import scipy.sparse as sp
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchtyping import TensorType
from typing import List, Tuple
from bidict import bidict
from sklearn.preprocessing import normalize
from collections import defaultdict

from trajectory_gnn.utils import graph_all_paths, markov_chain_compute_visit_probabilities
from trajectory_gnn.data.base import TrajectoryComplex, EdgeComplex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingBatch, TrajectoryForecastingPrediction, TrajectoryForecastingDataset
from trajectory_gnn.model.base import TrajectoryForecastingModelOneShotNoGradient
from trajectory_gnn.config import MarkovChainConfig

class TrajectoryMarkovChain(TrajectoryForecastingModelOneShotNoGradient):
    
    def __init__(self, config: MarkovChainConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
        self.order = config.order
        self.pseudo_count = config.pseudo_count
        self.cached = config.cached
        self.complex_probabilities_horizon = config.complex_probabilities_horizon
        self.non_backtracking = config.non_backtracking
        self.verbose = config.verbose
        if self.order < 1:
            raise ValueError(f'Invalid markov chain order {self.order}')
        
        edge_complex = dataset.base_dataset.edge_complex()
        
        paths = frozenset().union(*graph_all_paths(edge_complex.node_adjacency.T.detach().cpu().numpy(), self.order))
        self.history_to_idx = bidict({history : idx for idx, history in enumerate(set(h[:-1] for h in paths))})
        i, j = [], []
        if self.pseudo_count > 0:
            for history in paths:
                j.append(self.history_to_idx[history[:-1]])
                i.append(self.history_to_idx[history[1:]])
        self.transition_counts = sp.coo_matrix((np.ones(len(i), dtype=float) * self.pseudo_count, (i, j)), 
                                               shape=(len(self.history_to_idx), len(self.history_to_idx)))
        
        assert edge_complex.batch_size == 1, f'Should construct the markov chain from a non-collated single edge complex.' + \
            'Input complex contains {edge_complex.batch_size} graphs.'
        self.num_nodes = edge_complex.num_nodes
        self.num_states = len(self.history_to_idx)
        self.clear_cache()
        
    def clear_cache(self):
        self.transition_matrix_cache = None
        self.diffused_transition_matrix_cache = None
        self.visit_probabilties_cache = None
    
    @property
    def transition_matrix(self) -> sp.spmatrix:
        if self.transition_matrix_cache is not None:
            return self.transition_matrix_cache
        else:
            transition_matrix = normalize(self.transition_counts.tocsr(), norm='l1', axis=0)
            if self.cached:
                self.transition_matrix_cache = transition_matrix
            return transition_matrix
    
    @property
    def diffused_transition_matrix(self) -> sp.spmatrix:
        if self.diffused_transition_matrix_cache is not None:
            return self.diffused_transition_matrix_cache
        else:
            transition_matrix = self.transition_matrix
            diffused_transition_matrix = sp.eye(*transition_matrix.shape)
            for _ in range(self.complex_probabilities_horizon):
                diffused_transition_matrix = transition_matrix @ diffused_transition_matrix
            if self.cached:
                self.diffused_transition_matrix_cache = diffused_transition_matrix
            return diffused_transition_matrix
        
    @property
    def visit_probabilities(self) -> sp.spmatrix:
        """ Probabilities of visiting a node in `self.complex_probabilities_horizon` given a starting state. """
        if self.visit_probabilties_cache is not None:
            return self.visit_probabilties_cache
        else:
            transition_matrix = self.transition_matrix
            state_to_endpoint = np.array([self.history_to_idx.inverse[state_idx][-1] for state_idx in range(self.num_states)])
            visit_probabilities_cache = markov_chain_compute_visit_probabilities(transition_matrix,
                                                                                 self.complex_probabilities_horizon,
                                                                                 state_to_endpoint,
                                                                                 num_nodes=self.num_nodes,
                                                                                 verbose=self.verbose)
            if self.cached:
                self.visit_probabilties_cache = visit_probabilities_cache
            return visit_probabilities_cache
    
    def receptive_field(self) -> int | None:
        return 0 # note that we should never use subgraphs for the MC model though
    
    def fit_batch(self, batch: TrajectoryForecastingBatch, batch_idx: int):
        self.transition_counts = self.transition_counts.todok()
        counts_updated = False
        for xi in TrajectoryForecastingBatch.split(batch):
            assert xi.trajectory_complex.num_trajectories == 1, f'Markov chain does not yet support more than one trajectory per sample'
            for t_idx in range(xi.trajectory_complex.num_trajectories):
                edges = xi.trajectory_complex.edge_to_trajectory_adjacency[:, xi.trajectory_complex.edge_to_trajectory_adjacency[1] == t_idx][0]
                # concatenate the masked edges
                edge_idxs = torch.cat((edges, xi.masked_edge_idxs))
                edges = xi.trajectory_complex.edge_complex.node_adjacency[:, edge_idxs] # 2, trajectory_length
                edge_orientations = torch.cat((xi.trajectory_complex.edge_to_trajectory_orientation, xi.masked_edge_orientations))
                nodes = [edges[edge_orientations[0], 0].item()] + edges[1 - edge_orientations, torch.arange(edge_orientations.size(0))].tolist()
                for idx in range(len(nodes) - self.order):
                    history, successor = tuple(nodes[idx : idx + self.order]), tuple(nodes[idx + 1: idx + 1 + self.order])
                    if not self.non_backtracking or len(history) < 2 or history[-2] != successor:
                        self.transition_counts[self.history_to_idx[successor], self.history_to_idx[history]] += 1
                        counts_updated = True
                    
        if counts_updated:
            self.clear_cache() # The cache is invalid w.r.t. these counts
    
    @property
    def name(self) -> str:
        return f'markov_chain_{self.order}'
    
    def __call__(self, batch: TrajectoryComplex, context: str, *args) -> TrajectoryForecastingPrediction:
        """ Predicts one step using the counts of the markov model.

        Parameters
        ----------
        batch : TrajectoryComplex
            trajectory complex(es) to make predictions for
        context : str
            In which context ('train', 'val', 'test')

        Returns
        -------
        TrajectoryForecastingPrediction
            Prediction for this batch
        """
        transition_matrix = self.transition_matrix
        visit_probabilities = self.visit_probabilities
        device = batch.trajectory_batch_idxs.device
        
        complex_node_probs, step_node_probs = [], []
        for xi in TrajectoryComplex.split(batch):
            assert xi.num_trajectories == 1, f'Markov chain does not support more than one trajectory per sample'
            for t_idx in range(xi.num_trajectories):
                edges = xi.edge_to_trajectory_adjacency[:, xi.edge_to_trajectory_adjacency[1] == t_idx][0]
                edges = xi.edge_complex.node_adjacency[:, edges] # 2 x path_length
                edge_orientation = xi.edge_to_trajectory_orientation
                nodes = [edges[edge_orientation[0], 0].item()] + edges[1 - edge_orientation, torch.arange(edge_orientation.size(0))].tolist()
                history = tuple(nodes[-self.order:])
                
                state_step_probs = torch.from_numpy(transition_matrix[:, self.history_to_idx[history]].todense()
                                                         ).to(device).view(-1) # num_states,
                endpoints = torch.tensor([self.history_to_idx.inverse[idx][-1] for idx in range(self.num_states)],
                                         device=xi.trajectory_batch_idxs.device)
                step_node_probs.append(torch_scatter.scatter_add(state_step_probs, endpoints, dim_size=self.num_nodes))
                complex_node_probs.append(torch.from_numpy(visit_probabilities[:, self.history_to_idx[history]].todense()
                                                            ).to(device).view(-1))
        
        prediction = TrajectoryForecastingPrediction(
            probabilities=[torch.cat(complex_node_probs), None],
            )
        prediction.populate_successor_candidates_from_batch(batch)
        prediction.successor_candidate_node_probabilities = torch.cat(step_node_probs)[prediction.successor_candidate_node_idxs]
        return prediction
        
