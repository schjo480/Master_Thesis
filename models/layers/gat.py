# Layer that uses GCN-like simplicial complex convolutions

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn

from torchtyping import TensorType, patch_typeguard
from typing import List, Tuple
from typeguard import typechecked
import torch_scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


from trajectory_gnn.model.nn import InterOrderGCNConv, IntraOrderGCNConv, inter_order_gcn_norm
from trajectory_gnn.data.base import TrajectoryComplex
from .base import BaseLayer
from trajectory_gnn.utils.tensor_partition import TensorPartition

patch_typeguard()


class SCGATConvolution(nn.Module):
    
    def __init__(self, in_dim_src: int, in_dim_target: int, in_dim_meta_edge_features: int | None, out_dim: int,
                 num_heads: int = 8, leaky_relu_slope: float = .2, add_self_loops: bool=True, fill_value: float=0.0,
                 concat: bool=True, bias: bool=True, dropout: float=0.2,
                 ):
        super().__init__()
        
        self.num_heads = num_heads
        self.leaky_relu_slope = leaky_relu_slope
        self.add_self_loops = add_self_loops
        self.fill_value = fill_value
        self._out_dim = out_dim
        self.concat = concat
        self.bias = bias
        self.dropout = dropout
        
        self.emb_att = nn.Linear(in_dim_src + in_dim_target + (in_dim_meta_edge_features or 0), self.num_heads * self._out_dim)
        # Embedding for output values
        self.emb_target = nn.Linear(in_dim_target, self._out_dim)
        self.att = nn.Parameter(torch.Tensor(1, self.num_heads, self._out_dim))
        
        if self.bias and self.concat:
            self.bias = nn.Parameter(torch.Tensor(self.num_heads * self._out_dim))
        elif self.bias and not self.concat:
            self.bias = nn.Parameter(torch.Tensor(self._out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.emb_att.reset_parameters()
        self.emb_target.reset_parameters()
        glorot(self.att)
        zeros(self.bias)
        
    def forward(self, x_src: TensorType['num_src', 'num_src_features'], x_target: TensorType['num_target', 'num_target_features'],
                meta_edge_index: TensorType[2, 'num_meta_edges'], meta_edge_attr: TensorType['num_meta_edges', 'num_meta_edge_features'] | None) -> TensorType['num_src', 'out_dim']:
        # print('before', meta_edge_index.size(), meta_edge_attr.size() if meta_edge_attr is not None else None)
        if self.add_self_loops:
            assert x_src.size(0) == x_target.size(0), f'Self loops only make sense when source and target are on the same order.'
            num_nodes = x_src.size(0)
            meta_edge_index, meta_edge_attr = remove_self_loops(meta_edge_index, meta_edge_attr)
            meta_edge_index, meta_edge_attr = add_self_loops(meta_edge_index, meta_edge_attr,
                                                             fill_value=self.fill_value, num_nodes=num_nodes)

        if meta_edge_attr is not None and meta_edge_attr.dim() == 1:
            meta_edge_attr = meta_edge_attr.view(-1, 1)
            
        idx_src, idx_target = meta_edge_index[0], meta_edge_index[1]
        emb_input = [x_src[idx_src], x_target[idx_target]]
        if meta_edge_attr is not None:
            emb_input.append(meta_edge_attr)
            
        x = torch.cat(emb_input, dim=-1)
        # print(x.size(), [e.size() for e in emb_input], self.emb_att)
        
        x = self.emb_att(x)
        x = x.view(-1, self.num_heads, self._out_dim)
        x = F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        alpha = (x * self.att).sum(dim=-1) # -1 x num_heads
        alpha = torch_scatter.scatter_softmax(alpha, idx_src, dim=0, dim_size=x_src.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        values = self.emb_target(x_target)[idx_target].unsqueeze(-2) # -1, 1, out_dim
        propagated = torch_scatter.scatter_add(alpha.unsqueeze(-1) * values, idx_src, dim=0, dim_size=x_src.size(0)) # N, num_heads, out_dim
        # print(propagated.size())
        
        if self.concat:
            out = propagated.view(-1, self.num_heads * self._out_dim)
        else:
            out = propagated.mean(dim=1)
        
        if self.bias is not None:
            out += self.bias
        
        return out
    
    @property
    def out_dim(self) -> int:
        if self.concat:
            return self.num_heads * self._out_dim
        else:
            return self._out_dim

class SCGATLayer(BaseLayer):
    """ GAT convolution between all orders. """
    def __init__(self, 
                 in_dims: List[int | None],
                 out_dims: List[int | None],
                 add_self_loops: bool=True,
                 num_heads: int = 1,
                 leaky_relu_slope: float=0.2,
                 dropout: float=0.0,
                 fill_value: float = 0.0,
                 concat: bool=False,
                 **kwargs):
       
        super().__init__()
        
        self.concat = concat
        
        assert len(in_dims) == 3, len(out_dims) == 3
        self._out_dims = [[] for _ in range(3)]
        
        
        if out_dims[0]: # node
            if in_dims[0]:
                self.conv_0_0 = SCGATConvolution(in_dims[0], in_dims[0], in_dims[1], out_dims[0],
                                                 num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=add_self_loops, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[0].append(self.conv_0_0.out_dim)
            if in_dims[1]:
                self.conv_0_1 = SCGATConvolution(in_dims[0], in_dims[1], kwargs.get('in_dim_node_edge_adjacency', None),
                                                 out_dims[0], num_heads=num_heads, leaky_relu_slope=leaky_relu_slope, 
                                                 add_self_loops=False, fill_value=fill_value, 
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[0].append(self.conv_0_1.out_dim)
        
        if out_dims[1]: # edge
            if in_dims[1]:
                self.conv_1_1 = SCGATConvolution(in_dims[1], in_dims[1], in_dims[0], out_dims[1],
                                                 num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=add_self_loops, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[1].append(self.conv_1_1.out_dim)
            if in_dims[0]:
                self.conv_1_0 = SCGATConvolution(in_dims[1], in_dims[0], kwargs.get('in_dim_node_edge_adjacency', None), 
                                                 out_dims[1], num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=False, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[1].append(self.conv_1_0.out_dim)
            if in_dims[2]:
                self.conv_1_2 = SCGATConvolution(in_dims[1], in_dims[2], kwargs.get('in_dim_edge_trajectory_adjacency'), out_dims[1],
                                                 num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=False, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[1].append(self.conv_1_2.out_dim)
            
        if out_dims[2]: # trajectory (or any other order 2 cochain)
            if in_dims[2]:
                self.conv_2_2 = SCGATConvolution(in_dims[2], in_dims[2], kwargs.get('in_dim_trajectory_trajectory_adjacency', None),
                                                 out_dims[2], num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=add_self_loops, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[2].append(self.conv_2_2.out_dim)
            if in_dims[1]:
                self.conv_2_1 = SCGATConvolution(in_dims[2], in_dims[1], kwargs.get('in_dim_edge_trajectory_adjacency', None),
                                                 out_dims[2], num_heads=num_heads, leaky_relu_slope=leaky_relu_slope,
                                                 add_self_loops=False, fill_value=fill_value,
                                                 concat=concat, bias=True, dropout=dropout)
                self._out_dims[2].append(self.conv_2_1.out_dim)
        
    @property
    def out_dims(self) -> List[int | None]:
        if self.concat:
            out_dims = [sum(dims) for dims in self._out_dims]
        else:
            assert all(len(set(dims)) <= 1 for dims in self._out_dims)
            out_dims = [dims[0] if len(dims) > 0 else 0 for dims in self._out_dims]
        return out_dims
    
    @property
    def receptive_field(self) -> int | None:
        return 1
    
    def forward(self, x: List[TensorType | None],
                batch: TrajectoryComplex) -> List[TensorType | None]:
        outs = [[] for _ in range(len(x))]
        if hasattr(self, 'conv_0_0'):
            outs[0].append(self.conv_0_0(x[0], x[0], batch.edge_complex.node_adjacency, 
                                         x[1][batch.edge_complex.node_adjacency_edge_idxs]))
        if hasattr(self, 'conv_0_1'):
            outs[0].append(self.conv_0_1(x[0], x[1], batch.edge_complex.node_to_edge_adjacency,
                                         batch.edge_complex.node_to_edge_adjacency_features))
        if hasattr(self, 'conv_1_1'):
            outs[1].append(self.conv_1_1(x[1], x[1], batch.edge_complex.edge_adjacency,
                                         x[0][batch.edge_complex.edge_adjacency_node_idxs]))
        if hasattr(self, 'conv_1_0'):
            outs[1].append(self.conv_1_0(x[1], x[0], batch.edge_complex.node_to_edge_adjacency.flip(0),
                                         batch.edge_complex.node_to_edge_adjacency_features))
        if hasattr(self, 'conv_1_2'):
            outs[1].append(self.conv_1_2(x[1], x[2], batch.edge_to_trajectory_adjacency,
                                         batch.edge_to_trajectory_adjacency_features))
        if hasattr(self, 'conv_2_2'):
            outs[2].append(self.conv_2_2(x[2], x[2], batch.trajectory_adjacency,
                                         batch.trajectory_adjacency_features))
        if hasattr(self, 'conv_2_1'):
            outs[2].append(self.conv_2_1(x[2], x[1], batch.edge_to_trajectory_adjacency.flip(0),
                                         batch.edge_to_trajectory_adjacency_features))
            
        if self.concat:
            outs = [torch.cat(o, dim=-1) if len(o) > 0 else None for o in outs]
        else:
            outs = [torch.stack(o, dim=0).sum(dim=0) if len(o) > 0 else None for o in outs]
        return outs
        

class GATConv(BaseLayer):
    """ Only performs node-level aggregation using the standard GAT. """
    
    def __init__(self, in_dims: List[int], out_dims: List[int],
                 add_self_loops: bool=True, normalize: bool=True, node_dim: int=0,
                 num_heads: int = 1,
                 leaky_relu_slope: float=0.2,
                 dropout: float=0.0,
                 fill_value: float = 0.0,
                 concat: bool=False,
                 **kwargs):
        super().__init__()
        assert all(not d for d in out_dims[1:])
        self._out_dims = [d for d in out_dims]
        self.conv = tgnn.GATv2Conv(in_dims[0], out_dims[0], add_self_loops=add_self_loops,
                                   heads=num_heads, concat=concat, negative_slope=leaky_relu_slope,
                                   dropout=dropout, fill_value=fill_value)
    
    
    @property
    def receptive_field(self) -> int | None:
        return 1
        
    @property
    def out_dims(self) -> List[int]:
        return self._out_dims

    def forward(self, x: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        h_nodes = self.conv(x[0], batch.edge_complex.node_adjacency)
        return [h_nodes] + x[1:]      
        
                
        
    
        
        
        
        