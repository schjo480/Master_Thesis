import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torchtyping import TensorType
from typeguard import typechecked

from .base import BaseLayer
from trajectory_gnn.data.base import TrajectoryComplex

class Embedding(BaseLayer):
    """ Learns a node embedding for all nodes in the graph. """
    
    def __init__(self, in_dims: List[int], embedding_dims: List[int], *args, **kwargs):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.in_dims = in_dims
        
        if embedding_dims[0]:
            self.node_embedding = nn.Parameter(torch.zeros(kwargs['dataset'].num_nodes, embedding_dims[0]))
        else:
            self.node_embedding = None
            
        if embedding_dims[1]:
            self.edge_embedding = nn.Parameter(torch.zeros(kwargs['dataset'].num_edges, embedding_dims[0]))
        else:
            self.edges_embedding = None
    
    @property
    def out_dims(self) -> List[int]:
        return [in_dim + out_dim for in_dim, out_dim in zip(self.in_dims, self.embedding_dims)]
    
    @typechecked
    def forward(self, embeddings: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        out = [e for e in embeddings]
        if self.node_embedding is not None:
            emb_node = self.node_embedding[batch.edge_complex.node_idxs_to_uncollated_node_idxs(
                torch.arange(batch.edge_complex.num_nodes, device=batch.edge_to_trajectory_adjacency.device))]
            if out[0] is None:
                out[0] = emb_node
            else:
                out[0] = torch.cat((out[0], emb_node), dim=1)
        if self.edge_embedding is not None:
            emb_edge = self.edge_embedding[batch.edge_complex.edge_idxs_to_uncollated_edge_idxs(
                torch.arange(batch.edge_complex.num_edges, device=batch.edge_to_trajectory_adjacency.device))]
            if out[1] is None:
                out[1] = emb_edge
            else:
                out[1] = torch.cat((out[1], emb_edge), dim=1)
        return out
    
        