import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torchtyping import TensorType
from typeguard import typechecked

from .base import BaseLayer
from trajectory_gnn.data.trajectory_forecasting import TrajectoryComplex

class Dropout(BaseLayer):
    """ Dropout on all embeddings. """
    
    def __init__(self, in_dims: List[int], *args, **kwargs):
        super().__init__()
        self.in_dims = in_dims
        self.p = kwargs.get('p', 0.1)
        self.inplace = kwargs.get('inplace', False)
    
    @property
    def out_dims(self) -> List[int]:
        return self.in_dims
    
    @property
    def store_embedding(self) -> bool:
        return False # Dropout really isn't that interesting...
    
    @typechecked    
    def forward(self, embeddings: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        return [F.dropout(e, p=self.p, inplace=self.inplace, training=self.training) if e is not None else None for e in embeddings]