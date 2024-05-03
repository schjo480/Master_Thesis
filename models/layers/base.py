import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class BaseLayer(torch.nn.Module):
    """ Base layer class that takes embeddings of all orders and outputs embeddings of all orders. """
    
    @property
    def out_dims(self) -> List[int]:
        """ Output dimensions of all embeddings. """
        raise NotImplementedError
    
    @property
    def store_embedding(self) -> bool:
        """ Whether the embedding produced by this layer should be stored in the prediction. """
        return True
    
    @property
    def receptive_field(self) -> int | None:
        return 0