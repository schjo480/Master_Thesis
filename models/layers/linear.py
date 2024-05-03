import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torchtyping import TensorType
from typeguard import typechecked

from .base import BaseLayer
from trajectory_gnn.data.base import TrajectoryComplex

class Linear(BaseLayer):
    """ Linear transformation on all embeddings. """
    
    def __init__(self, in_dims: List[int], out_dims: List[int], *args, bias: bool=True, **kwargs):
        super().__init__()
        for in_dim, out_dim in zip(in_dims, out_dims):
            if out_dim and not in_dim:
                raise RuntimeError(f'Can not have empty inputs an non-empty outputs {in_dims}, {out_dims}')
        
        self._out_dims = out_dims
        self.linears = nn.ModuleDict({
            str(order) : nn.Linear(in_dim, out_dim, bias=bias)
            for order, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims))
            if in_dim and out_dim
        })
    
    @property
    def out_dims(self) -> List[int]:
        return self._out_dims
    
    @typechecked
    def forward(self, embeddings: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        return [self.linears[str(order)](e) if e is not None and str(order) in self.linears else None for order, e in enumerate(embeddings)]
        