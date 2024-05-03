from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch_scatter
import pytorch_lightning as pl
import torch_geometric.nn as tgnn
from omegaconf import DictConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from collections import defaultdict

from trajectory_gnn.model.base import TrajectoryForecasting
from trajectory_gnn.model.layers.gcn import SCGCNConv
from trajectory_gnn.data.base import TrajectoryComplex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingDataset, TrajectoryForecastingPrediction

patch_typeguard()

class TrajectoryForecastingSCNN(TrajectoryForecasting):
    """ Simplicial-like convolutional neural network for trajectory forecasting.

    Parameters
    ----------
    config : DictConfig
        The dict configuration.
    num_node_features : int
        How many node features the data has
    num_edge_features : int
        how many edge features the data has
    num_trajectory_features : int
        how many trajectory features the data has
    """
    
    def __init__(self, config: DictConfig, dataset: TrajectoryForecastingDataset):
        super().__init__()
        
        in_dims = (dataset.base_dataset.num_node_features, dataset.base_dataset.num_edge_features, 
                   dataset.base_dataset.num_trajectory_features)
        self.convs = nn.ModuleList()
        hidden_dims_nodes, hidden_dims_edges, hidden_dims_trajectories = config.hidden_dims_nodes, config.hidden_dims_edges, config.hidden_dims_trajectories
        
        # Build convolution layers
        for out_dims in zip(hidden_dims_nodes, hidden_dims_edges, hidden_dims_trajectories):
            conv = SCGCNConv(in_dims, out_dims, add_self_loops=config.add_self_loops, normalize=config.gcn_normalize, cached=config.cached, reduce=config.reduce)
            in_dims = conv.out_dims
            self.convs.append(conv)
        self.output = nn.ModuleDict()
        for order, (in_dim, out_dim) in enumerate(zip(in_dims, config.output_dims)):
            if out_dim is not None:
                self.output[str(order)] = nn.Linear(in_dim, out_dim)
        
    def forward(self, batch: TrajectoryComplex, context: str) -> TrajectoryForecastingPrediction:
        """ Forward pass.

        Parameters
        ----------
        batch : TrajectoryComplex
            Input trajectory complex.
        context : str
            In which context ('train', 'val', 'test')

        Returns
        -------
        result : TrajectoryForecastingPrediction
            The prediction result.
        """
        h = [batch.edge_complex.node_features, batch.edge_complex.edge_features, batch.trajectory_features]
        adj_intra = [batch.edge_complex.node_adjacency, batch.edge_complex.edge_adjacency, batch.trajectory_adjacency]
        adj_up = [batch.edge_complex.node_to_edge_adjacency, batch.edge_to_trajectory_adjacency]
        adj_down = [idxs.flip(0) for idxs in adj_up]
        embeddings = []
        
        for conv in self.convs:
            h = conv(h, adj_intra, adj_up, adjacency_down=adj_down)
            embeddings.append(h)
            h = [torch.relu(hi) if hi is not None else None for hi in h]
            
        embeddings.append([self.output[str(idx)](hi) if str(idx) in self.output else None for (idx, hi) in enumerate(h)])
        pred = TrajectoryForecastingPrediction(embeddings=embeddings)
        return pred
    