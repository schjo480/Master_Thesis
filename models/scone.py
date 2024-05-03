# Implementation of SCoNe for trajectory prediction

from typing import Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from omegaconf import DictConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


from models.layers import * # import all such that hydra can instantiate
from models.base import TrajectoryForecasting
from data.base import BaseDataset, CellComplex
from data.trajectory_forecasting import TrajectoryForecastingDataset, TrajectoryForecastingPrediction
from config import SCoNeConfig

patch_typeguard

class SCoNeConvolution(nn.Module):
    """ Simplicial convolution layer. """

    def __init__(self, in_dim: int, out_dim: int, activation: str='tanh'):
        super().__init__()
        self._activation = activation
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, 3*out_dim)
        
    @property
    def activation(self):
        activations = {
            'sigmoid' : torch.sigmoid,
            'relu' : torch.relu,
            'tanh' : torch.tanh,
        }
        if self._activation not in activations:
            raise ValueError(f'Unsupported activation for SCoNeLayer {self._activation}. Only {list(activations.keys())} are supported.')
        else:
            return activations[self._activation]
    
    @typechecked
    def _laplacian_convolution(self, x: TensorType['num_edges', 'd'], adjacency: TensorType[torch.long, 2, 'num_adj'], 
                               weight: TensorType['num_adj']) -> TensorType['num_edges', 'd']:
        idx_src, idx_target = adjacency
        message = x[idx_src] * weight.view(-1, 1)
        return torch_scatter.scatter_add(message, idx_target, dim=0, dim_size=x.size(0))
        
    @typechecked
    def forward(self, h_edges: TensorType['num_edges', 'in_dim'], batch: CellComplex) -> TensorType['num_edges', 'out_dim']:
        h_edges = self.lin(h_edges)
        h_lower = h_edges[..., : self.out_dim]
        h_intra = h_edges[..., self.out_dim : 2 * self.out_dim]
        h_upper = h_edges[..., 2 * self.out_dim :]
        h_lower = self._laplacian_convolution(h_lower, batch.edge_laplacian_lower_idxs, batch.edge_laplacian_lower_weights)
        h_upper = self._laplacian_convolution(h_upper, batch.edge_laplacian_upper_idxs, batch.edge_laplacian_upper_weights)
        result = self.activation(h_lower + h_intra + h_upper)
        return result

class SCoNe(TrajectoryForecasting):
    """ SCoNe model for trajectory prediction: https://arxiv.org/pdf/2102.10058v3.pdf"""
    def __init__(self, config: SCoNeConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
        self.input_features = config.input_features
        self.layers = nn.ModuleList([])
        
        in_dim = self.get_num_edge_input_features(dataset)
        for hidden_dim in config.hidden_dims:
            self.layers.append(SCoNeConvolution(in_dim, hidden_dim, activation=config.activation))
            in_dim = hidden_dim
        self.output = nn.Linear(in_dim, 1)
    
    def get_num_edge_input_features(self, dataset: TrajectoryForecastingDataset) -> int:
        if self.input_features == 'trajectory':
            in_dim = 1
        elif self.input_features == 'features':
            in_dim = self.num_edge_features(dataset) + self.num_appended_edge_features
        elif self.input_features == 'both':
            in_dim = self.num_edge_features(dataset) + 1 + self.num_appended_edge_features
        else:
            raise ValueError(f'Unsupported edge input features type {self.input_features}')
        return in_dim
            
    @typechecked
    def edge_chain_to_node_chain(self, batch: TrajectoryComplex, edge_signal: TensorType['num_edges', 'n']) -> TensorType['num_nodes', 'n']:
        """ Uses the boundary operator to downproject from edge signals to node signals using the orientation as a sign. """
        idx_target, idx_source = batch.edge_complex.node_to_edge_adjacency
        weights = batch.edge_complex.node_to_edge_orientation
        message = edge_signal[idx_source] * weights.view(-1, 1)
        node_signal = torch_scatter.scatter_add(message, idx_target, dim=0, dim_size=batch.edge_complex.num_nodes)
        return node_signal
        
    @typechecked
    def prepare_inputs(self, batch: TrajectoryComplex) -> Tuple[List[TensorType | None], TrajectoryComplex]:
        x, batch = super().prepare_inputs(batch)
        x_edges = x[1]
        if self.input_features in ('trajectory', 'both'):
            trajectory_indicator = torch.zeros(x_edges.size(0), 1, dtype=x_edges.dtype, device=x_edges.device)
            # `edge_to_trajectory_orientation` has 0 for in direction and 1 for flipped
            # in ScONe, 1 is in direction and -1 against direction : -> y = -2x + 1 to translate
            trajectory_indicator[batch.edge_to_trajectory_adjacency[0], 0] = (-2 * batch.edge_to_trajectory_orientation.float()) + 1 
        if self.input_features == 'both':
            x_edges = torch.cat((x_edges, trajectory_indicator), dim=-1)
        elif self.input_features == 'trajectory':
            x_edges = trajectory_indicator
        elif self.input_features == 'features':
            x_edges = x_edges
        else:
            raise ValueError(f'Unsupported edge input features type {self.input_features}')
        x[1] = x_edges
        return x, batch
            
    def forward(self, batch: CellComplex, context: str) -> TrajectoryForecastingPrediction:
        x, batch = self.prepare_inputs(batch)
        h_edge = x[1]
        embeddings = [[None, h_edge]]
        for layer in self.layers:
            h_edge = layer(h_edge, batch)
            embeddings.append([None, h_edge])
        out = self.output(h_edge)
        out_node = self.edge_chain_to_node_chain(batch, out)
        embeddings.append([out_node, out])
        return TrajectoryForecastingPrediction(embeddings=embeddings)
        
    @property
    def name(self) -> str:
        return f'scone_{len(self.layers)}'
        