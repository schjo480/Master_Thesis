# Pooling layers

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torchtyping import TensorType
from typeguard import typechecked

from .base import BaseLayer
from trajectory_gnn.data.base import TrajectoryComplex

class GlobalPooling(BaseLayer):
    """ Global pooling layer per order that is combined with each embedding. Also allows to combine pooled feature vectors
    across different orders. """
    
    POOLING_OPERATORS = ('mean', 'max')
    REDUCES = ('concat', 'sum', 'mean',)
    
    def __init__(self, in_dims: List[int], *args, cross_order: bool=False, which: str='max', reduce: str='concat', **kwargs):
        super().__init__()
        self.in_dims = list(in_dims)
        self.cross_order = cross_order
        if which not in self.POOLING_OPERATORS:
            raise ValueError(f'Unsupported pooling type {which}. Only {self.POOLING_OPERATORS} are supported.')
        self.which = which
        if reduce not in self.REDUCES:
            raise ValueError(f'Unsupported aggregation type {reduce}. Only {self.REDUCES} are supported.')
        if reduce in ('sum', 'mean') and self.cross_order and not len(set(in_dims)) == 1:
            raise ValueError(f'Can only aggregate pooled embeddings across orders if they all have the same size, not {self.in_dims}')
        self.reduce = reduce
    
    @property
    def out_dims(self) -> List[int]:
        if self.cross_order:
            dims = [[dim] + self.in_dims for dim in self.in_dims]
        else:
            dims = [[dim, dim] for dim in self.in_dims]
        if self.reduce in ('sum', 'mean'):
            return [d[0] for d in dims]
        else:
            return [sum((d for d in ds if d is not None), start=0) for ds in dims]
    
    @typechecked
    def forward(self, embeddings: List[TensorType | None], batch: TrajectoryComplex
                ) -> List[TensorType | None]:
        # Extract pooled representations
        pooled = []
        for e in embeddings:
            if e is None:
                pooled.append(None)
            else:
                if self.which == 'max':
                    pooled.append(e.max(dim=0, keepdim=True)[0])
                elif self.which == 'mean':
                    pooled.append(e.mean(dim=0, keepdim=True))
                else:
                    raise RuntimeError()
            
        if self.cross_order:
            # Every embedding is combined with pooled vectors of all orders
            others = [[p for p in pooled if p is not None] for _ in pooled]
        else:
            # Every embedding is only combined with pooled vector of the same order
            others = [[p] if p is not None else [] for p in pooled]
            
        if self.reduce == 'concat':
            return [torch.cat([e] + [p.expand(e.size(0), -1) for p in others_e], dim=1)
                    if e is not None else None 
                    for e, others_e in zip(embeddings, others)]
        elif self.reduce == 'sum':
            return [torch.stack([e] + [p.expand(e.size(0), -1) for p in others_e], dim=-1).sum(dim=-1)
                    if e is not None else None
                    for e, others_e in zip(embeddings, others)]
        elif self.reduce == 'mean':
            return [torch.stack([e] + [p.expand(e.size(0), -1) for p in others_e], dim=-1).mean(dim=-1)
                    if e is not None else None
                    for e, others_e in zip(embeddings, others)]
        else:
            raise RuntimeError()
            
    @property
    def receptive_field(self) -> int | None:
        return None # infinite receptive field
        
        
        