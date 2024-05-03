from typing import List, Tuple
from torchtyping import TensorType
from typeguard import typechecked

import torch
import torch_scatter
import attrs

from trajectory_gnn.data.base import Graph
from trajectory_gnn.data.simplicial_complex import collate_complex, split_complex

@attrs.define(frozen=True)
class EdgeComplex:
    """ Storage class for simplicial edge complexes (order 1). """
    node_features: TensorType['num_nodes', 'node_dim']
    edge_features: TensorType['num_edges', 'edge_dim']
    node_adjacency: TensorType[2, 'num_node_adj']
    node_adjacency_edge_idxs: TensorType['num_node_adj'] # via which edge two nodes are adjacenct
    node_to_edge_adjacency: TensorType[2, 'num_node_to_edge_adj']
    node_to_edge_orientation: TensorType['num_node_to_edge_adj'] # 1 is the start point, -1 the endpoint
    edge_adjacency: TensorType[2, 'num_edge_adj']
    edge_adjacency_node_idxs: TensorType['num_edge_adj'] # via which node to edges are adjacent
    
    # Helpers for associating node adjacency with edge features
    graphs: List[Graph]
    
    # Features for propagation
    node_adjacency_features: TensorType['num_node_adj', 'num_node_adj_features'] | None = None
    edge_adjacency_features: TensorType['num_edge_adj', 'num_edge_adj_features'] | None = None
    node_to_edge_adjacency_features: TensorType['num_node_to_edge_adj', 'num_node_to_edge_adj_features'] | None = None
    
    # Batching
    node_batch_idxs: TensorType['num_nodes'] = None
    edge_batch_idxs: TensorType['num_edges'] = None
    node_offsets: TensorType['batch_size'] = None
    edge_offsets: TensorType['batch_size'] = None
    batch_size: int = 1
    
    # For undirected complexes, there is less edges than entries in `node_adjacency`
    # This pointer keeps track of finding the correct index, i.e. maps edges
    # the actual node pair they refer to
    node_adjacency_sizes: List[int] = None
    
    def __attrs_post_init__(self):
        if self.node_batch_idxs is None:
            object.__setattr__(self, 'node_batch_idxs', torch.zeros(self.num_nodes, dtype=int, device=self.node_adjacency.device))
        if self.edge_batch_idxs is None:
            object.__setattr__(self, 'edge_batch_idxs', torch.zeros(self.num_edges, dtype=int, device=self.node_to_edge_adjacency.device))
        if self.node_adjacency_sizes is None:
            object.__setattr__(self, 'node_adjacency_sizes', [self.node_adjacency.size(1)])
            
        # Precompute by how many nodes each batch is offset
        if self.node_offsets is None:
            num_nodes_per_batch = torch_scatter.scatter(torch.ones_like(self.node_batch_idxs, dtype=int), 
                                            self.node_batch_idxs, dim=0, reduce='sum', dim_size=self.batch_size)
            object.__setattr__(self, 'node_offsets', torch.cumsum(num_nodes_per_batch, 0) - num_nodes_per_batch)
        if self.edge_offsets is None:
            num_edges_per_batch = torch_scatter.scatter(torch.ones_like(self.edge_batch_idxs, dtype=int), 
                                            self.edge_batch_idxs, dim=0, reduce='sum', dim_size=self.batch_size)
            object.__setattr__(self, 'edge_offsets', torch.cumsum(num_edges_per_batch, 0) - num_edges_per_batch)
            
    @property
    @torch.no_grad()
    def non_backtracking_random_walk_edge_adjacency_idxs(self):
        """ All indices in `edge_adjacency` that describe non-backtracking random walk, i.e. this removes all 2-cycles. """
        edge_idx_src, edge_idx_dst = self.edge_adjacency
        # for each edge adj (i.e. edge between edges) compute the node-endpoints of each edge-endpoint
        is_directed = torch.tensor([g.directed for g in self.graphs], dtype=bool, device=self.node_batch_idxs.device)
        edge_adjacency_batch_idxs = self.node_batch_idxs[self.edge_adjacency_node_idxs] # one possible way to map edge adjacency indices to batch indices
        edge_adjacency_is_directed = is_directed[edge_adjacency_batch_idxs]
        node_adjacency_src = self.node_adjacency[:, self.edge_idxs_to_node_adjacency_idxs(edge_idx_src)] # 2, num_edge_adj
        node_adjacency_dst = self.node_adjacency[:, self.edge_idxs_to_node_adjacency_idxs(edge_idx_dst)] # 2, num_edge_adj
        mask_backtracking = (node_adjacency_src.flip(0) == node_adjacency_dst).all(0) # num_edge_adj
        # in the case of directed complexes, we only want a "forward" adjacency
        # i.e. (a, b) <-> (b, c) but not (a, b) <-> (c, a)
        mask_forward = node_adjacency_src[1] == node_adjacency_dst[0] 
        return (~mask_backtracking) & (mask_forward | (~edge_adjacency_is_directed))
    
    @property
    def edge_adjacency_idx_to_node_adjacency_idx(self) -> TensorType['num_edge_adjacency']:
        """ Translates indices in the edge adjacency (e1 = (u, v) -> e2 = (v, w)) to indices in the node adjacency (v -> w)
        """
        sizes = torch.tensor(self.node_adjacency_sizes)
        node_adjacency_offsets = torch.cumsum(sizes, 0) - sizes
        return torch.cat([g.edge_adjacency_idx_to_node_adjacency_idx + offset 
                          for g, offset in zip(self.graphs, node_adjacency_offsets.tolist())]).to(self.node_batch_idxs.device)
        
    @property
    def observed_nodes(self) -> TensorType['num_observed_nodes']:
        return torch.cat([graph.observed_nodes.to(offset.device) + offset for graph, offset in zip(self.graphs, self.node_offsets)])
    
    @property
    def observed_edges(self) -> TensorType['num_observed_edges']:
        return torch.cat([graph.observed_edges.to(offset.device) + offset for graph, offset in zip(self.graphs, self.edge_offsets)])
    
    @property
    def num_nodes(self) -> int:
        if self.node_features is None: 
            if self.node_adjacency.numel() == 0:
                return 0
            return self.node_adjacency.max().item() + 1
        else:
            return self.node_features.size(0)
        
    @property
    def num_edges(self) -> int:
        if self.edge_features is None:
            return max(
                self.edge_adjacency.max().item() if self.edge_adjacency.numel() > 0 else -1, 
                self.node_to_edge_adjacency[1].max().item() if self.node_to_edge_adjacency.numel() > 0 else -1,
                -1) + 1
        else:
            return self.edge_features.size(0)

    def edge_idxs_to_node_adjacency_idxs(self, edge_idxs: TensorType['num_edge_idxs']) -> TensorType['num_edge_idxs']:
        """ Translates edge indices to indices in `self.node_adjacency`.
        """
        batch_idx = self.edge_batch_idxs[edge_idxs]
        
        # First, translate each edge_idx to the edge_idx on the uncollated complex
        edge_idxs = edge_idxs - self.edge_offsets[batch_idx]
        # Second, shift each edge_idx by the corresponding node_adjacency offset
        num_node_adjacencies = torch.tensor(self.node_adjacency_sizes, device=edge_idxs.device)
        node_adjacency_offsets = torch.cumsum(num_node_adjacencies, 0) - num_node_adjacencies
        return edge_idxs + node_adjacency_offsets[batch_idx]
        
    def successors_by_node(self, node_idx: int) -> List[Tuple[int, Tuple[int, int]]]:
        node_offsets, edge_offsets = self.node_offsets.tolist(), self.edge_offsets.tolist()
        node_batch_idx = self.node_batch_idxs[node_idx]
        return [(node_next + node_offsets[node_batch_idx], (edge_next + edge_offsets[node_batch_idx], edge_orientation_next)) for (node_next, (edge_next, edge_orientation_next)) in 
                  self.graphs[node_batch_idx].successors_by_node(node_idx - node_offsets[node_batch_idx])]
        
    def successors_by_edge(self, edge_idx: int, edge_orientation: int) -> List[Tuple[int, Tuple[int, int]]]:
        node_offsets, edge_offsets = self.node_offsets.tolist(), self.edge_offsets.tolist()
        edge_batch_idx = self.edge_batch_idxs[edge_idx]
        return [(node_next + node_offsets[edge_batch_idx], (edge_next + edge_offsets[edge_batch_idx], edge_orientation_next)) for (node_next, (edge_next, edge_orientation_next)) in 
                  self.graphs[edge_batch_idx].successors_by_edge(edge_idx - edge_offsets[edge_batch_idx], edge_orientation)]
        
    def oriented_edge_idxs_to_edge_idxs(self, edge_idxs: TensorType['num_edges'], edge_orientations: TensorType['num_edges'], 
                                        directed: bool) -> Tuple[TensorType['num_edges'], TensorType['num_edges']]:
        """ Transforms oriented edges into edge idxs on this complex. """
        if directed: 
            assert self.edge_features.size(0) % 2 == 0
            num_original_edges = self.edge_features.size(0) // 2 # each edge was used to create edges for both directions
            # Translate the orientation to the adequate edge idx, the orientation changes to 0 as each edge has direction information
            return edge_idxs + num_original_edges * (edge_orientations == 1), torch.zeros_like(edge_idxs)
        else:
            return edge_idxs, edge_orientations
    
    def node_distance(self, node_idxs_first: TensorType['num_distances'], node_idxs_second: TensorType['num_distances'],
                                    metric: str='sp_hops') -> TensorType['num_distances']:
        """Computes distance between two sets of nodes on this complex. """
        batch_idxs_first = self.node_batch_idxs[node_idxs_first]
        batch_idxs_second = self.node_batch_idxs[node_idxs_second]
        node_offsets = self.node_offsets.tolist()
        assert (batch_idxs_first == batch_idxs_second).all(), f'Shortest paths can only be calculated within one complex and not accross'
        return torch.tensor([
            self.graphs[b1].node_distance_metrics[metric][
                v1 - node_offsets[b1],
                v2 - node_offsets[b2]]
            for v1, b1, v2, b2 in zip(
                node_idxs_first.tolist(), 
                batch_idxs_first.tolist(),
                node_idxs_second.tolist(),
                batch_idxs_second.tolist())
        ], device=node_idxs_first.device)
    
    @typechecked
    def node_idxs_to_uncollated_node_idxs(self, node_idxs: TensorType[...]) -> TensorType[...]:
        """ Gets the uncollated node indices on the original complexes (after e.g. `self.split`) from node indices on the collated complex `self`.

        Parameters
        ----------
        node_idxs: TensorType[...]
            node indices on the collated complex (`self`)

        Returns
        -------
        TensorType[...]
            node indices on the uncollated complexes.
        """
        return node_idxs - self.node_offsets[self.node_batch_idxs[node_idxs]]
    
    @typechecked
    def edge_idxs_to_uncollated_edge_idxs(self, edge_idxs: TensorType[...]) -> TensorType[...]:
        """ Gets the uncollated edge indices on the original complexes (after e.g. `self.split`) from edge indices on the collated complex `self`.

        Parameters
        ----------
        edge_idxs: TensorType[...]
            Edge indices on the collated complex (`self`)

        Returns
        -------
        TensorType[...]
            Edge indices on the uncollated complexes.
        """
        return edge_idxs - self.edge_offsets[self.edge_batch_idxs[edge_idxs]]
    
    
    def _map_to_all_tensors(self, _func):
        object.__setattr__(self, 'node_adjacency', _func(self.node_adjacency))
        object.__setattr__(self, 'node_adjacency_edge_idxs', _func(self.node_adjacency_edge_idxs))
        object.__setattr__(self, 'node_to_edge_adjacency', _func(self.node_to_edge_adjacency))
        object.__setattr__(self, 'edge_adjacency', _func(self.edge_adjacency))
        object.__setattr__(self, 'edge_adjacency_node_idxs', _func(self.edge_adjacency_node_idxs))
        object.__setattr__(self, 'node_batch_idxs', _func(self.node_batch_idxs))
        object.__setattr__(self, 'edge_batch_idxs', _func(self.edge_batch_idxs))
        object.__setattr__(self, 'node_offsets', _func(self.node_offsets))
        object.__setattr__(self, 'edge_offsets', _func(self.edge_offsets))
        object.__setattr__(self, 'node_to_edge_orientation', _func(self.node_to_edge_orientation))
        if self.node_features is not None:
            object.__setattr__(self, 'node_features', _func(self.node_features))
        if self.edge_features is not None:
            object.__setattr__(self, 'edge_features', _func(self.edge_features))
        if self.node_to_edge_adjacency_features is not None:
            object.__setattr__(self, 'node_to_edge_adjacency_features', _func(self.node_to_edge_adjacency_features))
        
    def to(self, device) -> 'EdgeComplex':
        self._map_to_all_tensors(lambda tensor: tensor.to(device))
        return self
    
    def pin_memory(self) -> 'EdgeComplex':
        self._map_to_all_tensors(lambda tensor: tensor.pin_memory())
        return self
    
    
    @staticmethod
    def collate_edge_adjacency_node_idxs(batch: List['EdgeComplex']) -> TensorType:
        """ Collates which node connects an edge pair in `edge_adjacency` """
        node_batch_offsets = torch.tensor([b.num_nodes for b in batch], dtype=int)
        node_batch_offsets = torch.cumsum(node_batch_offsets, 0) - node_batch_offsets
        edge_adjacency_node_idxs = torch.cat([b.edge_adjacency_node_idxs + offset for b, offset in zip(batch, node_batch_offsets.tolist())])
        return edge_adjacency_node_idxs
    
    @staticmethod
    def collate_node_adjacency_edge_idxs(batch: List['EdgeComplex']) -> TensorType:
        """ Collates which edge connects an node pair in `node_adjacency` """
        edge_batch_offsets = torch.tensor([b.num_edges for b in batch], dtype=int, device=batch[0].edge_batch_idxs.device)
        edge_batch_offsets = torch.cumsum(edge_batch_offsets, 0) - edge_batch_offsets
        node_adjacency_edge_idxs = torch.cat([b.node_adjacency_edge_idxs + offset for b, offset in zip(batch, edge_batch_offsets.tolist())])
        return node_adjacency_edge_idxs
    
    @staticmethod
    def collate(batch: List['EdgeComplex']) -> 'EdgeComplex':
        if len(batch) == 1:
            return batch[0]
        
        (node_features, edge_features), (node_adjacency, edge_adjacency), (node_adjacency_features, edge_adjacency_features), (node_to_edge_adjacency, ), \
            (node_to_edge_orientation, ), (node_to_edge_adjacency_features, ), (node_batch_idxs, edge_batch_idxs) = \
            collate_complex(*zip(*[(
                [ec.node_features, ec.edge_features], 
                [ec.node_adjacency, ec.edge_adjacency],
                [ec.node_adjacency_features, ec.edge_adjacency_features],
                [ec.node_to_edge_adjacency], 
                [ec.node_to_edge_orientation],
                [ec.node_to_edge_adjacency_features],
                [ec.node_batch_idxs, ec.edge_batch_idxs]) for ec in batch]))
        
        return EdgeComplex(node_features=node_features, edge_features=edge_features, 
                           node_adjacency=node_adjacency, 
                           node_adjacency_edge_idxs=EdgeComplex.collate_node_adjacency_edge_idxs(batch),
                           edge_adjacency=edge_adjacency,
                           edge_adjacency_node_idxs=EdgeComplex.collate_edge_adjacency_node_idxs(batch),
                           node_to_edge_adjacency=node_to_edge_adjacency, node_batch_idxs=node_batch_idxs, edge_batch_idxs=edge_batch_idxs, 
                           batch_size=sum(b.batch_size for b in batch),
                           node_to_edge_orientation=node_to_edge_orientation,
                           graphs=sum((b.graphs for b in batch), start=[]),
                           node_adjacency_sizes=sum((b.node_adjacency_sizes for b in batch), start=[]),
                           node_adjacency_features=node_adjacency_features,
                           edge_adjacency_features=edge_adjacency_features,
                           node_to_edge_adjacency_features=node_to_edge_adjacency_features,
                           )
        
    @staticmethod
    def split_node_adjacency_edge_idxs(batch: 'EdgeComplex') -> List[TensorType]:
        """ Splits which edge connects a node pair in `node_adjacency` """
        num_edges_per_batch = torch_scatter.scatter(torch.ones_like(batch.edge_batch_idxs, dtype=int), 
                                        batch.edge_batch_idxs, dim=0, reduce='sum', dim_size=batch.batch_size)
        offsets_end = torch.cumsum(num_edges_per_batch, 0)
        offsets = offsets_end - num_edges_per_batch
        result = []
        for offset_start, offset_end in zip(offsets, offsets_end):
            mask = (batch.node_adjacency_edge_idxs >= offset_start) & (batch.node_adjacency_edge_idxs < offset_end)
            result.append(batch.node_adjacency_edge_idxs[mask] - offset_start)
        return result
    
    @staticmethod
    def split_edge_adjacency_node_idxs(batch: 'EdgeComplex') -> List[TensorType]:
        """ Splits which node connects an edge pair in `edge_adjacency` """
        num_nodes_per_batch = torch_scatter.scatter(torch.ones_like(batch.node_batch_idxs, dtype=int), 
                                        batch.node_batch_idxs, dim=0, reduce='sum', dim_size=batch.batch_size)
        offsets_end = torch.cumsum(num_nodes_per_batch, 0)
        offsets = offsets_end - num_nodes_per_batch
        result = []
        for offset_start, offset_end in zip(offsets, offsets_end):
            mask = (batch.edge_adjacency_node_idxs >= offset_start) & (batch.edge_adjacency_node_idxs < offset_end)
            result.append(batch.edge_adjacency_node_idxs[mask] - offset_start)
        return result
        
    @staticmethod
    def split(batch: 'EdgeComplex') -> List['EdgeComplex']:
        """ Inverse of `collate` """
        if batch.batch_size == 1:
            return [batch]
        features, adj_intra, features_adj_intra, adj_upper, adj_upper_orientation, features_adj_upper = split_complex(
            [batch.node_features, batch.edge_features],
            [batch.node_adjacency, batch.edge_adjacency],
            [batch.node_adjacency_features, batch.edge_adjacency_features],
            [batch.node_to_edge_adjacency],
            [batch.node_to_edge_orientation],
            [batch.node_to_edge_adjacency_features],
            [batch.node_batch_idxs, batch.edge_batch_idxs])
        
        return [EdgeComplex(node_features=node_features, edge_features=edge_features,
                                                           node_adjacency=adj_nodes_intra,
                                                           node_adjacency_features=features_adj_nodes_intra,
                                                           edge_adjacency_features=features_adj_edges_intra,
                                                           node_to_edge_adjacency=adj_node_to_edge,
                                                           node_to_edge_orientation=adj_node_to_edge_orientation,
                                                           node_to_edge_adjacency_features=features_adj_node_to_edge,
                                                           edge_adjacency=adj_edges_intra,
                                                           graphs=[graph],
                                                           node_adjacency_sizes=[node_adj_size],
                                                           node_adjacency_edge_idxs=node_adj_edge_idxs,
                                                           edge_adjacency_node_idxs=edge_adj_node_idxs,
                                                           ) for 
                ((node_features, edge_features), (adj_nodes_intra, adj_edges_intra), (features_adj_nodes_intra, features_adj_edges_intra), (adj_node_to_edge, ), \
                    (adj_node_to_edge_orientation, ), (features_adj_node_to_edge, ), graph, node_adj_size, node_adj_edge_idxs, edge_adj_node_idxs) in \
                    zip(features, adj_intra, features_adj_intra, adj_upper, adj_upper_orientation, features_adj_upper, batch.graphs, batch.node_adjacency_sizes,
                        EdgeComplex.split_node_adjacency_edge_idxs(batch), EdgeComplex.split_edge_adjacency_node_idxs(batch))]      
    
    @staticmethod
    @typechecked
    def from_graph(graph: Graph, node_features: TensorType['num_nodes', 'num_node_features'] | None,
                   edge_features: TensorType['num_edges', 'num_edge_features'] | None):
        return EdgeComplex(
                node_features=node_features,
                edge_features=edge_features,
                node_adjacency=graph.node_adjacency,
                node_adjacency_edge_idxs=graph.node_adjacency_edge_idxs,
                node_to_edge_adjacency=graph.node_to_edge_adjacency,
                node_to_edge_orientation=graph.node_to_edge_orientation,
                node_to_edge_adjacency_features=graph.node_to_edge_adjacency_features,
                edge_adjacency=graph.edge_adjacency,
                edge_adjacency_node_idxs=graph.edge_adjacency_node_idxs,
                graphs=[graph]
        )
        
        
        
        # if graph.directed:
        #     return EdgeComplex(
        #         node_features=node_features,
        #         edge_features=edge_features.repeat(2, 1) if edge_features is not None else None,
        #         node_adjacency=graph.node_adjacency,
        #         node_adjacency_edge_idxs=graph.node_adjacency_edge_idxs,
        #         node_to_edge_adjacency=graph.node_to_edge_adjacency,
        #         node_to_edge_orientation=graph.node_to_edge_orientation,
        #         node_to_edge_adjacency_features=graph.node_to_edge_adjacency_features,
        #         edge_adjacency=graph.edge_adjacency,
        #         edge_adjacency_node_idxs=graph.edge_adjacency_node_idxs,
        #         graphs=[graph]
        #     )
        # else:
        #     return EdgeComplex(
        #         node_features=node_features,
        #         edge_features=edge_features,
        #         node_adjacency=graph.node_adjacency,
        #         node_adjacency_edge_idxs=graph.node_adjacency_edge_idxs,
        #         node_to_edge_adjacency=graph.node_to_edge_adjacency,
        #         node_to_edge_orientation=graph.node_to_edge_orientation,
        #         node_to_edge_adjacency_features=graph.node_to_edge_adjacency_features,
        #         edge_adjacency=graph.edge_adjacency,
        #         edge_adjacency_node_idxs=graph.edge_adjacency_node_idxs,
        #         graphs=[graph]
        # )
