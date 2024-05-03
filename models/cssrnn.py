# Implementation of SCoNe for trajectory prediction

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
from trajectory_gnn.data.simplicial_complex import split_on_complex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingDataset, TrajectoryForecastingPrediction
from trajectory_gnn.config import CSSRNNConfig

class LearnedEdgeEmbedding(nn.Module):
    """ Learned embedding for all edges. """
    
    def __init__(self, num_edges: int, embedding_dim: int):
        super().__init__()
        self.num_edges = num_edges
        self.embedding = nn.Embedding(self.num_edges, embedding_dim)
    
    @typechecked
    def forward(self, x: List[TensorType[float] | None], batch: TrajectoryComplex, 
                edge_idxs: TensorType['num_edges', int]) -> TensorType['num_edges', 'embedding_dim', float]:
        assert batch.edge_complex.num_edges == self.num_edges * batch.batch_size, f'Edge embeddings are pre-learned so the input batch graphs must exactly be the dataset graph'
        return self.embedding(batch.edge_complex.edge_idxs_to_uncollated_edge_idxs(edge_idxs))

class LinearEdgeEmbedding(nn.Module):
    """ Linearly transforms embeddings of edge input features. """ 
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, embedding_dim)
        
    @typechecked
    def forward(self, x: List[TensorType[float] | None], batch: TrajectoryComplex, 
                edge_idxs: TensorType['num_edges', int]) -> TensorType['num_edges', 'embedding_dim', float]:
        x_nodes = x[0]
        x_edges = x[1]
        
        endpoint_idxs = batch.edge_complex.node_adjacency[:, batch.edge_complex.edge_idxs_to_node_adjacency_idxs(edge_idxs)].T # edge_idxs.size(0), 2
        endpoint_features = x_nodes[endpoint_idxs] # edge_idxs.size(0), 2, num_node_dim
        endpoint_features = endpoint_features.reshape(edge_idxs.size(0), -1)
        
        return self.lin(torch.cat((x_edges[edge_idxs], endpoint_features), dim=-1))


class CSSRNN(TrajectoryForecasting):
    """ CSSRNN model for trajectory prediction: https://www.ijcai.org/proceedings/2017/0430.pdf 
    
    Uses an RNN (LSTM) to process the edge sequence and outputs a distribution over successor edges.
    Does not explicitly get any topological information (other than restricting the output distribution to successors)
    """
    
    valid_edge_embeddings = ('learned', 'linear', 'both')
    
    
    def __init__(self, config: CSSRNNConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
        self.num_layers = config.num_layers
        self.lstm = nn.LSTM(input_size=config.edge_embedding_size, hidden_size=config.hidden_dim, 
                                    num_layers=self.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        # The embedding of each trajectory is translated to logits on an edge by linear projection
        # This is equivalent to a dot product with learned embeddings (=linear projection)
        # This way, we do not need to evaluate on all edges in the graph (which can even be batched)
        self._initialize_embedding(config, dataset)
        
        
    def _initialize_embedding(self, config: CSSRNNConfig, dataset: TrajectoryForecastingDataset):
        if not config.edge_embedding in self.valid_edge_embeddings:
            raise ValueError(f'Edge embedding must be in {self.valid_edge_embeddings}')
        
        self.input_embeddings = nn.ModuleList()
        self.output_embeddings = nn.ModuleList()
        output_embedding_dim = config.hidden_dim * self.num_layers * (2 if config.bidirectional else 1)
        
        if config.edge_embedding in ('learned', 'both'):
            self.input_embeddings.append(LearnedEdgeEmbedding(dataset.num_edges, config.edge_embedding_size))
            self.output_embeddings.append(LearnedEdgeEmbedding(dataset.num_edges, output_embedding_dim))
        if config.edge_embedding in ('linear', 'both'):
            num_edge_input_features = self.num_edge_features(dataset)
            num_node_input_features = self.num_node_features(dataset)
            self.input_embeddings.append(LinearEdgeEmbedding(num_edge_input_features + 2 * num_node_input_features, config.edge_embedding_size))
            self.output_embeddings.append(LinearEdgeEmbedding(num_edge_input_features + 2 * num_node_input_features, output_embedding_dim))
    
    @property
    def receptive_field(self) -> int | None:
        return None
    
    @typechecked
    def get_input_embedding(self, x: List[TensorType[float] | None], batch: TrajectoryComplex, 
                edge_idxs: TensorType['num_edges', int]) -> TensorType['num_edges', 'embedding_dim', float]:
        return torch.stack([embedding(x, batch, edge_idxs) for embedding in self.input_embeddings]).sum(0)
    
    @typechecked
    def get_output_embedding(self, x: List[TensorType[float] | None], batch: TrajectoryComplex, 
                edge_idxs: TensorType['num_edges', int]) -> TensorType['num_edges', 'embedding_dim', float]:
        return torch.stack([embedding(x, batch, edge_idxs) for embedding in self.input_embeddings]).sum(0)
        
    
    @typechecked
    def forward(self, batch: TrajectoryComplex, context: str) -> TrajectoryForecastingPrediction:
        x, batch = self.prepare_inputs(batch)
        edge_idxs = batch.edge_to_trajectory_adjacency[0]
        emb_input = self.get_input_embedding(x, batch, edge_idxs)
        
        _, (h, _) = self.lstm(nn.utils.rnn.pack_sequence(split_on_complex(emb_input, batch.edge_complex.edge_batch_idxs[edge_idxs]),
                                                         enforce_sorted=False)) # D * num_layers, num_trajectories, hidden_dim
        h = h.transpose(0, 1)
        h = h.view(-1, h.size(1) * h.size(2)) # num_trajectories, hidden_dim * D * num_layers´
        pred = TrajectoryForecastingPrediction()
        pred.populate_successor_candidates_from_batch(batch)
        successor_candidate_edge_idxs_embeddings = self.get_output_embedding(x, batch, pred.successor_candidate_edge_idxs) # num_candidates, hidden_dim * D * num_layers
        
        # Only output logits for reachable edges from each trajectory, i.e. `successor_candidate_edge_idxs`
        # The logits are given as the (hidden state of each trajectory) x (a learned embedding for each (candidate) edge)
        successor_candidate_edge_logits = (h[pred.successor_candidate_trajectory_idxs] * successor_candidate_edge_idxs_embeddings).sum(-1)
        pred.successor_candidate_logits = [None, successor_candidate_edge_logits, None]
        return pred
    
    @typechecked         
    def sample_successors(self, batch: TrajectoryComplex, pred: TrajectoryForecastingPrediction) -> TensorType['num_trajectories']:
        """ Samples the successor indices from a data sample and the model predictions on it """
        # Default strategy: Sample the most likely node
        return pred.get_sampled_successor_candidate_idxs_by_max_score(by_order=1)
        
    @property
    def name(self) -> str:
        return f'cssrnn_{self.num_layers}'
        
