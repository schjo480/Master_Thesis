from typing import List
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


from trajectory_gnn.model.layers import * # import all such that hydra can instantiate
from trajectory_gnn.model.base import TrajectoryForecasting
from trajectory_gnn.data.base import BaseDataset, TrajectoryComplex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingDataset, TrajectoryForecastingPrediction
from trajectory_gnn.config import SequentialConfig


from hydra.utils import instantiate

patch_typeguard()

class TrajectoryForecastingSequential(TrajectoryForecasting):
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
    
    
    def __init__(self, config: SequentialConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
        in_dims = (self.num_node_features(dataset),
                   self.num_edge_features(dataset), 
                   dataset.base_dataset.num_trajectory_features)
        
        kwargs = {
            'in_dim_node_edge_adjacency' : dataset.num_node_to_edge_adjacency_features,
            'in_dim_edge_trajectory_adjacency' : dataset.num_edge_to_trajectory_adjacency_features + \
                self.num_appended_edge_to_trajectory_adjacency_features,
            'dataset' : dataset,
        }
        
        self.layers = nn.ModuleList()
        for layer in config.layers:
            layer: BaseLayer = instantiate(layer, in_dims, **kwargs)
            in_dims = layer.out_dims
            self.layers.append(layer)
    
    @typechecked    
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
        h, batch = self.prepare_inputs(batch)
        embeddings = [h]
        for layer in self.layers:
            h = layer(h, batch)
            if layer.store_embedding:
                embeddings.append(h)
        pred = TrajectoryForecastingPrediction(embeddings=embeddings)
        return pred
    
    @property
    def receptive_field(self) -> int | None:
        receptive_field = 0
        for layer in self.layers:
            layer: BaseLayer = layer
            layer_receptive_field = layer.receptive_field
            if layer_receptive_field is None:
                return None
            else:
                receptive_field += layer_receptive_field
        return receptive_field
    
    @property
    def name(self) -> str:
        return f'sequential_{len(self.layers)}'
        