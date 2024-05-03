# Utility modules that implement various parameterless convolution types

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch_scatter
import torch_geometric as tg
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def inter_order_gcn_norm(edge_index: TensorType[2, 'num_edges'], 
                         edge_weight: Optional[TensorType['num_edges']],
                         dtype=None) -> Tuple[TensorType[2, 'num_edges'], TensorType['num_edges']]:
    """ Performs symmetric GCN normalization on inter order adjacencies. That is, 
        w^hat_ij = deg_out(i)^-1/2 * w_ij * deg_in(j)^-1/2.
        
        This is akin to the GCN conv for a normal adjacency, which normalizes w_ij according to
        w^hat_ij = deg_in(i)^-1/2 * w_ij * deg_in(j)^-1/2
        Note that in the inter-order gcn norm, we use the out degree of the source node and not its in-degree.
        This is, because the "connection graph" between the simplices of different order is bipartite and all
        source nodes have in degree 0. 
        The formulation this convolution implements is equivalent to a normal GCN normalization on the
        undirected, bipartite "connection graph" between simplices of different order without self-loops.
        There, in degree and out degree are equal.
        """
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                    device=edge_index.device)
    idx_source, idx_target = edge_index[0], edge_index[1]
    num_nodes_source = int(idx_source.max()) + 1 if idx_source.numel() > 0 else 0
    num_nodes_target = int(idx_target.max()) + 1 if idx_target.numel() > 0 else 0
    
    deg_inv_sqrt_source = torch.zeros(num_nodes_source, dtype=dtype).to(edge_weight.device).scatter_(0, idx_source, edge_weight, reduce='add').pow_(-0.5)
    deg_inv_sqrt_source.masked_fill_(deg_inv_sqrt_source == float('inf'), 0)
    deg_inv_sqrt_target = torch.zeros(num_nodes_target, dtype=dtype).to(edge_weight.device).scatter_(0, idx_target, edge_weight, reduce='add').pow_(-0.5)
    deg_inv_sqrt_target.masked_fill_(deg_inv_sqrt_target == float('inf'), 0)
    edge_weight = deg_inv_sqrt_source[idx_source] * edge_weight * deg_inv_sqrt_target[idx_target]
    return edge_index, edge_weight

class InterOrderGCNConv(nn.Module):
    """ Convolution between simplices of different order using the GCN framework. """

    def __init__(self, cached: bool=True, normalize: bool=True, node_dim: int=0, reduce: str='sum'):
        super().__init__()
        self.cached = cached
        self.normalize = normalize
        self.node_dim = node_dim
        self._cache = None
        self.reduce = reduce
        
    @typechecked
    def forward(self, x: TensorType['num_simplices_in', 'hidden_dim'], 
                edge_index: TensorType[2, 'num_edges'], # idx_source, idx_target
                edge_weight: Optional[TensorType['num_edges']] = None,
                num_nodes_target: int | None = None) -> TensorType['num_simplices_out', 'hidden_dim']:
        if self.normalize:
            if self._cache is None:
                edge_index, edge_weight = inter_order_gcn_norm(edge_index, edge_weight, dtype=x.dtype)
                if self.cached:
                    self._cache = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = self._cache
        
        idx_source, idx_target = edge_index[0], edge_index[1]
        if num_nodes_target is None:
            num_nodes_target = int(idx_target.max()) + 1 if idx_target.numel() > 0 else 0
        message = x[idx_source] * edge_weight.view(-1, 1) if edge_weight is not None else x[idx_source]
        # use `torch_scatter.scatter` instead of `torch.scatter_` as it allows for a more flexible reduction operator (min, max, ...)
        aggregated = torch_scatter.scatter(message, idx_target, dim=0, dim_size=num_nodes_target, reduce=self.reduce)
        return aggregated
        

class IntraOrderGCNConv(tgnn.MessagePassing):
    """ Convolution between simplices of the same order. Boils down to a GCN conv. """
    
    def __init__(self, add_self_loops: bool=True, cached: bool=True, normalize: bool=True, node_dim: int=0):
        super().__init__()
        self.add_self_loops = add_self_loops
        self.cached = cached
        self.normalize = normalize
        self.node_dim = node_dim
        self._cache = None
        
    @typechecked
    def forward(self, x: TensorType['num_simplices', 'hidden_dim'], 
                edge_index: TensorType[2, 'num_edges'], 
                edge_weight: Optional[TensorType['num_edges']] = None) -> TensorType['num_simplices', 'hidden_dim']:
        """ Forward pass. """
        
        if self.normalize:
            if self._cache is None:
                edge_index, edge_weight = gcn_norm(edge_index, edge_weight, 
                                                   num_nodes = x.size(self.node_dim), add_self_loops=self.add_self_loops, 
                                                   dtype=x.dtype)
                if self.cached:
                    self._cache = (edge_index, edge_weight)
            else:
                edge_index, edge_weight = self._cache

        return self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
    
    @typechecked
    def message(self, x_j: TensorType['num_messages', 'message_dim'], edge_weight: Optional[TensorType['num_messages']]) -> TensorType['num_messages', 'message_dim']:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
            