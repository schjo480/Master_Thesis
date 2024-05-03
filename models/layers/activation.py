import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torchtyping import TensorType
from typeguard import typechecked

from .base import BaseLayer
from trajectory_gnn.data.trajectory_forecasting import TrajectoryComplex

class ReLU(BaseLayer):
    """ ReLU activation function on all embeddings. """
    
    def __init__(self, in_dims: List[int], *args, **kwargs):
        super().__init__()
        self.in_dims = in_dims
        self.inplace = kwargs.get('inplace', False)
    
    @property
    def out_dims(self) -> List[int]:
        return self.in_dims
    
    @property
    def store_embedding(self) -> bool:
        return False # ReLU really isn't that interesting...
    
    @typechecked    
    def forward(self, embeddings: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        return [F.relu(e, self.inplace) if e is not None else None for e in embeddings]