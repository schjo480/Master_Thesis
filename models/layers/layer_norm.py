from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtyping import TensorType

from trajectory_gnn.model.layers.base import BaseLayer
from trajectory_gnn.data.base import TrajectoryComplex

class LayerNorm(BaseLayer):
    """ Applies layer norm, i.e. normalization along the feature dimension. """
    
    def __init__(self, in_dims: List[int],
                 eps: float = 1e-5, elementwise_affine: bool=True, **kwargs):
        super().__init__()
        
        self._out_dims = [d for d in in_dims]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.ParameterDict()
        self.bias = nn.ParameterDict()
        if self.elementwise_affine:
            for order, dim in enumerate(in_dims):
                self.weight[str(order)] = nn.Parameter(torch.empty(dim))
                self.bias[str(order)] = nn.Parameter(torch.empty(dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        for param in self.weight.values():
            torch.nn.init.ones_(param)
        for param in self.bias.values():
            torch.nn.init.zeros_(param)
            
    def forward(self, x: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        out = []
        for order, xi in enumerate(x):
            out.append(F.layer_norm(xi, (xi.size(-1),), self.weight.get(str(order), None), 
                                    self.bias.get(str(order)), eps=self.eps))
        return out
            
    @property
    def out_dims(self) -> List[int]:
        return self._out_dims
        