from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn

from torchtyping import TensorType
from hydra.utils import instantiate

from trajectory_gnn.model.layers.base import BaseLayer
from trajectory_gnn.data.base import TrajectoryComplex

class ResidualLayer(BaseLayer):
    """ Wraps a layer with residual connections """
    
    def __init__(self,
                 in_dims: List[int | None],
                 layers: DictConfig,
                 concat: bool=False,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        out_dims = in_dims
        for layer in layers:
            layer: BaseLayer = instantiate(layer, out_dims, **kwargs)
            out_dims = layer.out_dims
            self.layers.append(layer)
        
        self.concat = concat
        
        self._out_dims = []
        self.projection = nn.ModuleDict()
        
        for order, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            if in_dim and out_dim:
                if self.concat:
                    self._out_dims.append(in_dim + out_dim)
                else:
                    if in_dim != out_dim:
                        self.projection[str(order)] = nn.Linear(in_dim, out_dim)
                    self._out_dims.append(out_dim)
            else:
                self._out_dims.append(0)
    
    def forward(self, x: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        h = x
        for layer in self.layers:
            h = layer(h, batch)
        out = []
        for order, (xi, hi )in enumerate(zip(x, h)):
            if xi is not None and hi is not None:
                if self.concat:
                    out.append(torch.cat((xi, hi), dim=-1))
                else:  
                    if str(order) in self.projection:
                        xi = self.projection[str(order)](xi)
                    out.append(xi + hi)
            else:
                out.append(None)
        return out
            
    @property
    def out_dims(self) -> List[int]:
        return self._out_dims
        
        
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
        
        
        
        
        