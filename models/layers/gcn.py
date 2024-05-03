# Layer that uses GCN-like simplicial complex convolutions

from collections import defaultdict
import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

from torchtyping import TensorType, patch_typeguard
from typing import List, Tuple
from typeguard import typechecked
import torch_scatter
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from trajectory_gnn.model.nn import InterOrderGCNConv, IntraOrderGCNConv, inter_order_gcn_norm
from trajectory_gnn.data.base import TrajectoryComplex
from .base import BaseLayer

patch_typeguard()

class GCNConv(BaseLayer):
    """ Only performs node-level aggregation using the standard GCN. """
    
    def __init__(self, in_dims: List[int], out_dims: List[int],
                 add_self_loops: bool=True, normalize: bool=True, node_dim: int=0,
                 **kwargs):
        super().__init__()
        assert all(not d for d in out_dims[1:])
        self._out_dims = [d for d in out_dims]
        self.conv = tgnn.GCNConv(in_dims[0], out_dims[0], add_self_loops=add_self_loops,
                                 normalize=normalize, cached=False)
    
    @property
    def receptive_field(self) -> int | None:
        return 1
    
    @property
    def out_dims(self) -> List[int]:
        return self._out_dims

    def forward(self, x: List[TensorType | None], batch: TrajectoryComplex) -> List[TensorType | None]:
        h_nodes = self.conv(x[0], batch.edge_complex.node_adjacency)
        return [h_nodes] + x[1:]


class SCGCNConv(BaseLayer):
    """ Module for a GCN-like convolution defined on simplicial-like complexes. 

        Parameters
        ----------
        in_dims : List[int]
            Number of input features per co-chain order.
        out_dims : List[int  |  None]
            Number of output features per co-chain order. If None is given, no output is computed for this cochain.
        add_self_loops : bool, optional
            For convolutions from co-chain of order k to oder k (intra-order), if self-loops are added, by default True.
        normalize : bool, optional
            If the adjacency matrices for both intra- and inter-order convolutions are normalized, by default True
        cached : bool, optional
            If convolution matrices should be cached, by default True
        node_dim : int, optional
            Which axis contains the co-chain dimension, by default 0
        reduce : str, optional
            How to aggregate informatino from intra- and inter-convolutions, by default 'concat'
        """
    
    def __init__(self, 
                 in_dims: List[int],
                 out_dims: List[int | None],
                 add_self_loops: bool=True,
                 normalize: bool=True,
                 cached: bool=False, # Caching is a bad idea when batching (the input complex changes)
                 node_dim: int=0,
                 reduce: str='concat',
                 **kwargs):
       
        super().__init__()
        
        self.in_dims = in_dims
        self._out_dims = out_dims
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self.node_dim = node_dim
        self.reduce = reduce
        self.order = len(self.in_dims)
        
        # Linear operators (concatenated for speedup) and convolution operators
        # Even though convolution operators are parameterless, they can cache information
        # so we ininialize them separately
        self.emb = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        for idx in range(self.order):
            in_dim = self.in_dims[idx]
            out_dim = self._out_dims[idx]
            if out_dim is not None:
                self.conv[str((idx, idx))] = IntraOrderGCNConv(add_self_loops=self.add_self_loops, cached=self.cached, normalize=self.normalize, node_dim=self.node_dim)
            else:
                out_dim = 0 # No output for this k-chain order, but there still is information flow to orders below and above (possibly)
            if idx > 0 and self._out_dims[idx - 1] is not None: 
                out_dim += self._out_dims[idx - 1]  # Information to order below
                self.conv[str((idx, idx - 1))] = InterOrderGCNConv(cached=self.cached, normalize=self.normalize, node_dim=self.node_dim)
            if idx < self.order - 1 and self._out_dims[idx + 1] is not None:
                out_dim += self._out_dims[idx + 1] # Information to order above
                self.conv[str((idx, idx + 1))] = InterOrderGCNConv(cached=self.cached, normalize=self.normalize, node_dim=self.node_dim)
            self.emb[str(idx)] = nn.Linear(in_dim, out_dim)
    
    @property
    def receptive_field(self) -> int | None:
        return 1
    
    @property
    def out_dims(self) -> List[int]:
        if self.reduce in ('concat',):
            out_dims = []
            for order, out_dim in enumerate(self._out_dims):
                if out_dim is None:
                    out_dims.append(out_dim)
                elif order in range(1, self.order - 1):
                    out_dims.append(3 * out_dim)
                else:
                    out_dims.append(2 * out_dim) # Either no information flow from below or above
            return out_dims
        else:
            return [out_dim for out_dim in self._out_dims]
    
    @typechecked
    def forward(self,
                x: List[TensorType],
                batch: TrajectoryComplex, 
                ) -> List[TensorType | None]:
        """ Forward pass.
        
        Parameters:
        -----------
        x : List[TensorType]
            The representation of all simplices. Each element corresponds to one order.
        adjacency_intra : List[TensorType[2, -1]]
            The adjacency for simplices of a given order. Each element is a tensor of shape [2, num_connections] 
            that is to be interpreted like an "edge" between simplices.
        adjacency_weight_intra : Optional[List[TensorType[-1]]]
            Weights for `adjacency_intra`
        adjacency_up : List[TensorType[2, -1]]
            The adjacency between simplices of a given order to the ones of one order higher. Each element is of
            shape [2, num_connections] that is to be interpreted like an "edge" between simplices.
        adjacency_weight_up : Optional[List[TensorType[-1]]]
            Weights for `adjacency_up`
        adjacency_down : Optional[TensorType[2, -1]]
            The adjacency between simplices of a given order to the ones of one order lower. Each element is of
            shape [2, num_connections] that is to be interpreted like an "edge" between simplices.
            If None, `adjacency_up` is simply transposed.
        adjacency_weight_down : Optional[List[TensorType[-1]]]
            Weights for `adjacency_down`
        
        Returns:
        --------
        h : List[TensorType]
            Representation of all simplices after the layer.
        """
        adjacency_intra = [batch.edge_complex.node_adjacency, batch.edge_complex.edge_adjacency, batch.trajectory_adjacency]
        adjacency_up = [batch.edge_complex.node_to_edge_adjacency, batch.edge_to_trajectory_adjacency]
        adjacency_down = [idx.flip(0) for idx in adjacency_up]
        
        adjacency_weight_intra = [None for _ in adjacency_intra]
        adjacency_weight_down = [None for _ in adjacency_down]
        adjacency_weight_up = [None for _ in adjacency_up]
        
        # Linear transformation
        hidden = []
        for idx in range(self.order):
            h = self.emb[str(idx)](x[idx])
            # each hidden state contains embeddings for intra, lower and upper adjacency
            if self._out_dims[idx] is not None:
                h_intra = h[..., : self._out_dims[idx]]
                offset = self._out_dims[idx]
            else:
                h_intra = None
                offset = 0
            if idx > 0 and self._out_dims[idx - 1] is not None:
                h_lower = h[..., offset : offset + self._out_dims[idx - 1]]
                offset += self._out_dims[idx - 1]
            else:
                h_lower = None
            if idx < self.order - 1 and self._out_dims[idx + 1] is not None:
                h_upper = h[..., offset : offset + self._out_dims[idx + 1]]
                offset += self._out_dims[idx + 1]
            else:
                h_upper = None
            hidden.append((h_intra, h_lower, h_upper))
        
        # Convolutions
        result = []
        for idx in range(self.order):
            num_of_order = [hi.size(0) for hi in hidden[idx] if hi is not None][0] # infer the number of "node / edge / trajectory"
            result_idx = []
            # Intra-order convolution
            if str((idx, idx)) in self.conv:
                result_idx.append(self.conv[str((idx, idx))](hidden[idx][0], adjacency_intra[idx], adjacency_weight_intra[idx]))
            if str(((idx - 1), idx)) in self.conv:
                result_idx.append(self.conv[str((idx - 1, idx))](hidden[idx - 1][2], adjacency_up[idx - 1], edge_weight=adjacency_weight_up[idx - 1],
                                                                 num_nodes_target=num_of_order))
            if str(((idx + 1), idx)) in self.conv:
                result_idx.append(self.conv[str((idx + 1, idx))](hidden[idx + 1][1], adjacency_down[idx], edge_weight=adjacency_weight_down[idx],
                                                                 num_nodes_target=num_of_order))
            if len(result_idx) > 0:
                # Aggregate
                if self.reduce in ('concat',):
                    result.append(torch.cat(result_idx, -1))
                elif self.reduce in ('max',):
                    result.append(torch.stack(result_idx, 0).max()[0])
                elif self.reduce in ('mean',):
                    result.append(torch.stack(result_idx, 0).mean()[0])
                else:
                    raise RuntimeError(f'Unsupported reduction type {self.reduce}')
            else:
                result.append(None)
            
        for out_dim, r in zip(self.out_dims, result):
            if out_dim is None:
                assert r is None
            else:
                assert r.size(-1) == out_dim
        
        return result
    
class MessageSCGNConv(BaseLayer):
    
    """ Module for a GCN-like convolution defined on simplicial-like complexes using learnable meassges between layers.

        Parameters
        ----------
        in_dims : List[int]
            Number of input features per co-chain order.
        out_dims : List[int  |  None]
            Number of output features per co-chain order. If None is given, no output is computed for this cochain.
        add_self_loops : bool, optional
            For convolutions from co-chain of order k to oder k (intra-order), if self-loops are added, by default True.
        normalize : bool, optional
            If the adjacency matrices for both intra- and inter-order convolutions are normalized, by default True
        cached : bool, optional
            If convolution matrices should be cached, by default True
        node_dim : int, optional
            Which axis contains the co-chain dimension, by default 0
        reduce : str, optional
            How to aggregate informatino from intra- and inter-convolutions, by default 'concat'
        inter_order_message_dims : List[int] | None
            Dimensionalities of messages from order k -> k + 1 (and k + 1 -> k)
        intra_order_message_dims : List[int] | None
            Dimensionalities of messages from order k->k
        """
    
    def __init__(self, 
                 in_dims: List[int],
                 out_dims: List[int | None],
                 add_self_loops: bool=True,
                 normalize: bool=True,
                 cached: bool=False, # Caching is a bad idea when batching (the input complex changes)
                 node_dim: int=0,
                 reduce: str='concat',
                 inter_order_message_dims: List[int | None] | None = None,
                 intra_order_message_dims: List[int | None] | None = None):
       
        super().__init__()
        
        self.in_dims = in_dims
        self._out_dims = out_dims
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.cached = cached
        self.node_dim = node_dim
        self.reduce = reduce
        self.order = len(self.in_dims)
        self.information_flow_between = defaultdict(bool)
        self.inter_order_message_dims = inter_order_message_dims or [None] * (self.order - 1)
        self.intra_order_message_dims = intra_order_message_dims or [None] * self.order
        
        # Linear operators (concatenated for speedup) and convolution operators
        # We aggregate linear transformations to all flows
        self.emb = nn.ModuleDict()
        self.message = nn.ModuleDict()
        for idx in range(self.order):
            in_dim = self.in_dims[idx]
            if in_dim is None: # no information inflow can also not be propagated to other orders
                continue
            emb_out_dim = 0
            
            if self._out_dims[idx] is not None:
                if self.intra_order_message_dims[idx] is None:
                    emb_out_dim += self._out_dims[idx] # Transform inter-order
                else:
                    self.message[str((idx, idx))] = nn.Linear(2 * self.in_dims[idx] + self.intra_order_message_dims[idx], self._out_dims[idx])
                self.information_flow_between[(idx, idx)] = True
            
            if idx > 0 and self._out_dims[idx - 1] is not None:
                if self.inter_order_message_dims[idx - 1] is None:
                    emb_out_dim += self._out_dims[idx - 1]
                else:
                    self.message[str((idx, idx - 1))] = nn.Linear(self.in_dims[idx] + self.in_dims[idx - 1] + self.inter_order_message_dims[idx - 1])
                self.information_flow_between[(idx, idx - 1)] = True
            
            if idx < self.order - 1 and self._out_dims[idx + 1] is not None:
                if self.inter_order_message_dims[idx] is None:
                    emb_out_dim += self._out_dims[idx + 1]
                else:
                    self.message[str(idx, idx + 1)] = nn.Linear(self.in_dims[idx] + self.in_dims[idx + 1] + self.inter_order_message_dims[idx])
                self.information_flow_between[(idx, idx + 1)] = True
                
            if emb_out_dim > 0:
                self.emb[str(idx)] = nn.Linear(self.in_dims[idx], emb_out_dim)
    
    @property
    def receptive_field(self) -> int | None:
        return 1
    
    @property
    def out_dims(self) -> List[int]:
        if self.reduce in ('concat',):
            out_dims = []
            for order, out_dim in enumerate(self._out_dims):
                if out_dim is None:
                    out_dims.append(None)
                elif order in range(1, self.order - 1):
                    out_dims.append(3 * out_dim)
                else:
                    out_dims.append(2 * out_dim) # No information flow either form below or above
            return out_dims
        else:
            return [out_dim for out_dim in self._out_dims]
    
    def _transform_and_split(self, x: TensorType | None, order: int) -> Tuple[TensorType | None, 
                                                                    TensorType | None, 
                                                                    TensorType | None]:
        """ Transforms input features where"""
        if x is None:
            return None, None, None
        if str(order) in self.emb:
            h = self.emb[str(order)](x)
        else:
            h = None
        # split the embedding into parts that are propagated to intra-, up- and down-convs
        offset = 0
        
        # intra-order
        if self._out_dims[order] is None:
            h_intra = None
        else:
            if self.intra_order_message_dims[order] is None:
                 h_intra = h[..., offset : offset + self._out_dims[order]]
                 offset += self._out_dims[order]
            else:
                h_intra = x
        
        # down
        if order > 0 and self._out_dims[order - 1] is not None:
            if self.inter_order_message_dims[order - 1] is None:
                h_down = h[..., offset : offset + self._out_dims[order - 1]]
                offset += self._out_dims[order - 1]
            else:
                h_down = x
        else:
            h_down = None
            
        # up
        if order < self.order - 1 and self._out_dims[order + 1] is not None:
            if self.inter_order_message_dims[order] is None:
                h_up = h[..., offset : offset + self._out_dims[order + 1]]
                offset += self._out_dims[order + 1]
            else:
                h_up = x
        else:
            h_up = None
        
        return h_intra, h_down, h_up

    def _aggregate(self,
                   message: TensorType['num_messages', 'message_dim'],
                    adj: TensorType[2, 'num_adj'],
                    intra_order: bool,
                    adj_weight: TensorType['num_adj'] | None=None,
                    add_self_loops: bool=True,
                    normalize: bool=True,
                    num_nodes_target: int | None = None,) -> TensorType['num_nodes_target', 'message_dim']:
        if normalize:
            if intra_order:
                adj, adj_weight = gcn_norm(adj, edge_weight=adj_weight, num_nodes=num_nodes_target,
                                           add_self_loops=add_self_loops)
            else:
                adj, adj_weight = inter_order_gcn_norm(adj, adj_weight, dtype=message.dtype)

        if adj_weight is not None:
            message *= adj_weight.view(-1, 1)
        aggregated = torch_scatter.scatter(message, adj[1], dim=0, dim_size=num_nodes_target,
                                           reduce='sum')
        return aggregated
        

    @typechecked
    def forward(self,
                x: List[TensorType],
                batch: TrajectoryComplex, 
                ) -> List[TensorType | None]:
        adjacency_intra = [batch.edge_complex.node_adjacency, batch.edge_complex.edge_adjacency, batch.trajectory_adjacency]
        adjacency_up = [batch.edge_complex.node_to_edge_adjacency, batch.edge_to_trajectory_adjacency]
        adjacency_down = [idx.flip(0) for idx in adjacency_up]
        
        intra_order_adjacency_features = [batch.edge_complex.node_adjacency_features, batch.edge_complex.edge_adjacency_features, batch.trajectory_adjacency_features]
        up_adjacency_features = [batch.edge_complex.node_to_edge_adjacency_features, batch.edge_to_trajectory_adjacency_features]
        down_adjacency_features = up_adjacency_features
        
        adjacency_weight_intra = [None for _ in adjacency_intra]
        adjacency_weight_down = [None for _ in adjacency_down]
        adjacency_weight_up = [None for _ in adjacency_up]
        
        # Linear transformation
        hidden = [self._transform_and_split(x[order], order) for order in range(self.order)]
        
        result = []
        for order in range(self.order):
            result_order = []
            
            # intra-conv
            if self.information_flow_between[(order, order)]:
                adj, adj_weight = adjacency_intra[order], adjacency_weight_intra[order]
                if self.intra_order_message_dims[order] is None:
                    message = hidden[order][0][adj[0]]
                else:
                    message = self.message[(str(order))](torch.cat((
                        x[order][adj[0]], 
                        x[order][adj[1]], 
                        intra_order_adjacency_features[order]), dim=-1))
                result_order.append(self._aggregate(message, adj, True, adj_weight=adj_weight,
                                                    add_self_loops=self.add_self_loops, normalize=self.normalize,
                                                    num_nodes_target=x[order].size(0)))

            # up-conv
            if self.information_flow_between[(order - 1, order)]:
                adj, adj_weight = adjacency_up[order], adjacency_weight_up[order]
                if self.inter_order_message_dims[order - 1] is None:
                    message = hidden[order - 1][2][adj[0]]
                else:
                    message = self.message[(str(order - 1), str(order))](torch.cat((
                        x[order - 1][adj[0]],
                        x[order][adj[1]],
                        up_adjacency_features[order - 1],
                    ), dim=-1))
                result_order.append(self._aggregate(message, adj, False, adj_weight=adj_weight,
                                                    add_self_loops=False, normalize=self.normalize,
                                                    num_nodes_target=x[order].size(0)))
            
            # down-conv
            if self.information_flow_between[(order + 1, order)]:
                adj, adj_weight = adjacency_down[order], adjacency_weight_down[order]
                if self.inter_order_message_dims[order] is None:
                    message = hidden[order + 1][1][adj[0]]
                else:
                    message = self.message[(str(order + 1), str(order))](torch.cat((
                        x[order + 1][adj[0]],
                        x[order][adj[1]],
                        down_adjacency_features[order],
                    ), dim=-1))
                result_order.append(self._aggregate(message, adj, False, adj_weight=adj_weight,
                                                    add_self_loops=False, normalize=self.normalize,
                                                    num_nodes_target=x[order].size(0)))                
                    
            if len(result_order) > 0:
                result.append(torch.cat(result_order, dim=-1))
            else:
                result.append(None)
        return result  
            
            