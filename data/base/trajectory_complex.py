import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict
from torchtyping import TensorType
from typeguard import typechecked

import torch
import torch_scatter
import attrs

from trajectory_gnn.data.simplicial_complex import collate_complex, split_complex
from trajectory_gnn.data.base import EdgeComplex
from trajectory_gnn.utils.utils import gather_batch_idxs

@attrs.define(frozen=True)
class TrajectoryComplex:
    """ Storage class for a simplicial-like complex involving trajectories.
    Thos are input instances for trajectory models. """
    
    edge_complex: EdgeComplex
    trajectory_features: TensorType['num_trajectories', 'trajectory_dim'] | None
    edge_to_trajectory_adjacency: TensorType[2, 'num_edges_in_trajectories'] = attrs.field(
        validator=lambda tc, attr, val: val.size(1) == tc.edge_to_trajectory_orientation.size(0))
    trajectory_adjacency: TensorType[2, 'num_inter_trajectory_adj']
    edge_to_trajectory_orientation: TensorType['num_edges_in_trajectories'] = None # 0 means trajectory follows edge *in* direction, 1 means trajectory follows edge against it's direction
    edge_to_trajectory_adjacency_features: TensorType['num_edges_in_trajectories', 'num_edges_in_trajectories_features'] | None = None
    trajectory_adjacency_features: TensorType['num_inter_trajectory_adj', 'num_inter_trajectory_adj_features'] | None = None
    
    attributes: List[dict] = attrs.field(factory=lambda: [{}])
    # Batching
    trajectory_batch_idxs: TensorType['num_trajectories'] = None
    
    def __attrs_post_init__(self):
        if self.trajectory_batch_idxs is None:
            object.__setattr__(self, 'trajectory_batch_idxs', torch.zeros(self.num_trajectories, dtype=int, device=self.edge_to_trajectory_adjacency.device))
    
    def advance(self, edge_idxs: TensorType['num_advance'], 
                edge_orientations: TensorType['num_advance'],
                trajectory_idxs: TensorType['num_advance'],
                edge_to_trajectory_adjacency_features: TensorType['num_advance', 'num_edge_to_trajectory_adjacency_features'] | None = None,
                fill_value: float = 0.0) -> 'TrajectoryComplex':
        """Adds new edges to trajectories

        Parameters
        ----------
        edge_idxs : TensorType[&#39;num_advance&#39;]
            The edges to add. If multiple edges are added to one trajectory, they are expected to be in the right sequential order.
        edge_orientations : TensorType[&#39;num_advance&#39;]
            The orientation of `edge_idxs`
        trajectory_idxs : TensorType[&#39;num_advance&#39;]
            For each edge in `edge_idx`, to which trajectory to add it.
        edge_to_trajectory_adjacency_features : TensorType['num_advance', 'num_edge_to_trajectory_adjacency_features'], optional
            The features of the edge to trajectory adjacency to advance. If not given, will be padded with `fill_value`
        fill_value : float, default = 0.0
            If `edge_to_trajectory_adjacency_features` will be padded with this value.

        Returns
        -------
        TrajectoryComplex
            _description_
        """
        if self.edge_to_trajectory_adjacency_features is None:
            assert edge_to_trajectory_adjacency_features is None, 'Cant append edge_to_trajectory_adjacency_features to None'
        else:
            if edge_to_trajectory_adjacency_features is None:
                edge_to_trajectory_adjacency_features =  torch.full(
                    (edge_idxs.size(0), self.edge_to_trajectory_adjacency_features.size(1)),
                    fill_value=fill_value,
                    dtype=self.edge_to_trajectory_adjacency_features.dtype,
                    device=self.edge_to_trajectory_adjacency_features.device,
                    )
            edge_to_trajectory_adjacency_features = torch.cat((self.edge_to_trajectory_adjacency_features, edge_to_trajectory_adjacency_features), dim=0)
        
        return attrs.evolve(self,
                            edge_to_trajectory_orientation = torch.cat((self.edge_to_trajectory_orientation, edge_orientations)),
                            edge_to_trajectory_adjacency = torch.cat((self.edge_to_trajectory_adjacency, torch.stack((edge_idxs, trajectory_idxs))), dim=-1),
                            edge_to_trajectory_adjacency_features = edge_to_trajectory_adjacency_features,
                            )
        
    @typechecked
    def get_advance_successors(self) -> Tuple[TensorType['num_successors'], TensorType['num_successors'], TensorType['num_successors'], 
                                              TensorType['num_successors'], TensorType['num_nodes'], TensorType['num_edges']]:
        """ Returns neighbourhoods for the endpoints of all trajectories.

        Returns
        -------
        TensorType['num_successors']
            node indices of successors
        TensorType['num_successors']
            edge indices of successors
        TensorType['num_successors']
            edge orientations of successors
        TensorType['num_successors']
            which trajectory this is a successor of
        TensorType['num_nodes']
            A map of each node to the index in `node_successors`.
            If a node is not present,  it is assigned -1
        TensorType['num_edge']
            A map of each edge to the index in `edge_successors`.
            If an edge is not present,  it is assigned -1
        """
        nodes, edges, edge_orientations, trajectory_idxs = [], [], [], []
        for endpoint_edge, endpoint_edge_orientation, trajectory_idx in zip(*self.trajectory_endpoints):
            if endpoint_edge != -1 and endpoint_edge_orientation != -1:
                for node_idx, (edge_idx, edge_orientation) in self.edge_complex.successors_by_edge(endpoint_edge.item(), endpoint_edge_orientation.item()):
                    nodes.append(node_idx)
                    edges.append(edge_idx)
                    edge_orientations.append(edge_orientation)
                    trajectory_idxs.append(trajectory_idx)
        device = self.edge_complex.node_adjacency.device
        node_successors = torch.tensor(nodes, device=device, dtype=int)
        edge_successors = torch.tensor(edges, device=device, dtype=int)
        node_map = -torch.ones(self.edge_complex.num_nodes, device=device, dtype=int)
        node_map[node_successors] = torch.arange(node_successors.size(0), device=device)
        edge_map = -torch.ones(self.edge_complex.num_edges, device=device, dtype=int)
        edge_map[edge_successors] = torch.arange(edge_successors.size(0), device=device) 
        return node_successors, edge_successors, torch.tensor(edge_orientations, device=device), torch.tensor(trajectory_idxs, device=device), \
            node_map, edge_map
    
    @property
    def trajectory_lengths(self) -> TensorType['num_trajectories']:
        """ How many edges are in each trajectory. """
        return torch_scatter.scatter_sum(torch.ones(self.edge_to_trajectory_adjacency.size(1), device=self.edge_to_trajectory_adjacency.device, dtype=int),
                                         self.edge_to_trajectory_adjacency[1], dim_size=self.num_trajectories)
        
    @property
    def trajectory_node_idxs(self) -> Tuple[TensorType['num_nodes_in_all_trajectories'], TensorType['num_nodes_in_all_trajectories']]:
        """ All node idxs that are in the trajectories of this complex.
        
        Returns:
        --------
        TensorType['num_nodes_in_all_trajectories']
            The node indices (in order) of all trajectories
        TensorType['num_nodes_in_all_trajectories']
            Which trajectory index each of the nodes belongs to.
        """
        startpoint_edge_idxs, startpoint_edge_orientations, startpoint_trajectory_idxs = self.trajectory_startpoints
        edge_idxs = torch.cat([startpoint_edge_idxs, self.edge_to_trajectory_adjacency[0]])
        edge_orientations = torch.cat([1 - startpoint_edge_orientations, self.edge_to_trajectory_orientation])
        trajectory_idxs = torch.cat([startpoint_trajectory_idxs, self.edge_to_trajectory_adjacency[1]])
        node_adjacency_idxs = self.edge_complex.edge_idxs_to_node_adjacency_idxs(edge_idxs)
        node_idxs = self.edge_complex.node_adjacency[1 - edge_orientations, node_adjacency_idxs]
        return node_idxs, trajectory_idxs

    @property
    def trajectory_startpoints(self) -> Tuple[TensorType['num_nonempty_trajectories'], TensorType['num_nonempty_trajectories'], TensorType['num_nonempty_trajectories']]:
        """ Startpoints for each trajectory in the batch that is not empty. One does not need to call `oriented_edge_idxs_to_edge_idxs`, as this should be done when
        constructing the edge complex (and in fact `TrajectoryComplex.from_edge_complex` does that.)

        Returns
        -------
        TensorType['num_nonempty_trajectories']
            The edge indices that are the startpoints
        TensorType['num_nonempty_trajectories']
            The edge orientations that are the startpoints
        TensorType['num_nonempty_trajectories']
            The trajectory idx of each startpoint.
        """
        idxs = torch_scatter.scatter_min(torch.arange(self.edge_to_trajectory_orientation.size(0), dtype=int, device=self.edge_to_trajectory_adjacency.device), 
                                         self.edge_to_trajectory_adjacency[1], dim_size=self.num_trajectories,
                                         out=torch.full((self.num_trajectories,), self.edge_complex.num_edges, dtype=int, device=self.edge_to_trajectory_adjacency.device))[0]
        mask = idxs < self.edge_complex.num_edges # trajectories that are non-empty
        return self.edge_to_trajectory_adjacency[0, idxs[mask]], self.edge_to_trajectory_orientation[idxs[mask]], \
            torch.arange(self.num_trajectories, device=self.edge_to_trajectory_adjacency.device)[mask]
        
    @property
    def trajectory_endpoints(self) -> Tuple[TensorType['num_nonempty_trajectories'], TensorType['num_nonempty_trajectories'], TensorType['num_nonempty_trajectories']]:
        """ Endpoints for each trajectory in the batch that is not empty. One does not need to call `oriented_edge_idxs_to_edge_idxs`, as this should be done when
        constructing the edge complex (and in fact `TrajectoryComplex.from_edge_complex` does that.)

        Returns
        -------
        TensorType['num_nonempty_trajectories']
            The edge indices that are the endpoints
        TensorType['num_nonempty_trajectories']
            The edge orientations that are the endpoints
        TensorType['num_nonempty_trajectories']
            The trajectory idx of each endpoint.
        """
        idxs = torch_scatter.scatter_max(torch.arange(self.edge_to_trajectory_orientation.size(0), dtype=int, device=self.edge_to_trajectory_adjacency.device), 
                                         self.edge_to_trajectory_adjacency[1], dim_size=self.num_trajectories,
                                         out=torch.full((self.num_trajectories,), -1, dtype=int, device=self.edge_to_trajectory_adjacency.device))[0]
        mask = idxs >= 0 # trajectories that are non-empty
        return self.edge_to_trajectory_adjacency[0, idxs[mask]], self.edge_to_trajectory_orientation[idxs[mask]], \
            torch.arange(self.num_trajectories, device=self.edge_to_trajectory_adjacency.device)[mask]
            
    def distance_from_trajectory_endpoints(self, metric: str='sp_hops') -> Tuple[TensorType['num_nodes'] | None, TensorType['num_edges'] | None]:
        """ Computes the distance of all instances in the edge complex from the respective trajectory endpoints. """
        node_lengths, length_node_idxs = [], []
        edge_lengths, length_edge_idxs = [], []
        
        endpoint_edge_idxs, endpoint_edge_orientations, endpoint_trajectory_idxs = self.trajectory_endpoints
        endpoint_node_idxs = self.edge_complex.node_adjacency[1 - endpoint_edge_orientations, 
                                                                                 self.edge_complex.edge_idxs_to_node_adjacency_idxs(endpoint_edge_idxs)]
        endpoint_batch_idxs = self.trajectory_batch_idxs[endpoint_trajectory_idxs]
        node_offsets = self.edge_complex.node_offsets.tolist()
        edge_offsets = self.edge_complex.edge_offsets.tolist()
        
        for node_idx, batch_idx in zip(endpoint_node_idxs.tolist(), endpoint_batch_idxs.tolist()):
            graph = self.edge_complex.graphs[batch_idx]
            sp_lengths_nodes = graph.node_distance_metrics.get(metric, None)
            if sp_lengths_nodes is not None:
                sp_lengths_nodes_node_idx = sp_lengths_nodes[node_idx - node_offsets[batch_idx]]
                if graph.directed:
                    # the edge distance is the distance of the edges actual endpoint
                    sp_lenghts_edges = sp_lengths_nodes_node_idx[graph.node_adjacency[1]]
                else:
                    # the edge distance is whichever edge endpoint is closer
                    sp_lenghts_edges = sp_lengths_nodes_node_idx[graph.node_adjacency[: , : graph.num_edges]].min(0)[0] 
                
                node_lengths.append(sp_lengths_nodes_node_idx)
                edge_lengths.append(sp_lenghts_edges)
                length_node_idxs.append(torch.arange(sp_lengths_nodes_node_idx.size(0)) + node_offsets[batch_idx])
                length_edge_idxs.append(torch.arange(sp_lenghts_edges.size(0)) + edge_offsets[batch_idx])
            else:
                node_lengths.append(None)
                edge_lengths.append(None)
                length_node_idxs.append(None)
                length_edge_idxs.append(None)
            
        # The scatter min also handles if there's two trajectories in a single complex (which it shouldn't)
        # It takes the shorter path of the two. If certain nodes or edges are unreachable, they get inf
        if any(t is None for t in node_lengths):
            node_lengths = None
        else:
            node_lengths = torch_scatter.scatter(torch.cat(node_lengths), torch.cat(length_node_idxs), reduce='min', out=torch.full((self.edge_complex.num_nodes,), torch.inf))
            
        if any(t is None for t in edge_lengths):
            edge_lengths = None
        else:
            edge_lengths = torch_scatter.scatter(torch.cat(edge_lengths), torch.cat(length_edge_idxs), reduce='min', out=torch.full((self.edge_complex.num_edges,), torch.inf))
        return node_lengths, edge_lengths
    
    def _pad_edge_to_trajectory_idxs(self):
        order, mask = gather_batch_idxs(self.edge_to_trajectory_adjacency[1])
        object.__setattr__(self, '_padded_edge_to_trajectory_idxs', self.edge_to_trajectory_adjacency[0][order])
        object.__setattr__(self, '_padded_edge_to_trajectory_orientations', self.edge_to_trajectory_orientation[order])
        object.__setattr__(self, '_padded_edge_to_trajectory_idxs_mask', mask)
        
    @property
    def padded_edge_to_trajectory_idxs(self) -> TensorType['batch_size', 'max_num_edges_in_trajectory']:
        if not hasattr(self, '_padded_edge_to_trajectory_idxs'):
            self._pad_edge_to_trajectory_idxs()
        return self._padded_edge_to_trajectory_idxs
    
    @property
    def padded_edge_to_trajectory_orientations(self) -> TensorType['batch_size', 'max_num_edges_in_trajectory']:
        if not hasattr(self, '_padded_edge_to_trajectory_orientations'):
            self._pad_edge_to_trajectory_idxs()
        return self._padded_edge_to_trajectory_orientations
    
    @property
    def padded_edge_to_trajectory_idxs_mask(self) -> TensorType['batch_size', 'max_num_edges_in_trajectory']:
        if not hasattr(self, '_padded_edge_to_trajectory_idxs_mask'):
            self._pad_edge_to_trajectory_idxs()
        return self._padded_edge_to_trajectory_idxs_mask
    
    @property
    def num_trajectories(self) -> int:
        if self.trajectory_features is None:
            return max(self.trajectory_adjacency.max()[0], self.edge_to_trajectory_adjacency[1].max()[0]) + 1
        else:
            return self.trajectory_features.size(0)
        
    @property
    def num_trajectories_per_batch(self) -> TensorType['batch_size']:
        """ How many trajectories are in each batch. """
        return torch_scatter.scatter_add(torch.ones_like(self.trajectory_batch_idxs),
                                         self.trajectory_batch_idxs, dim_size=self.batch_size)
    
        
    
    @staticmethod
    def from_edge_complex(edge_complex: EdgeComplex, trajectory_edge_idxs: List[TensorType], 
                          trajectory_edge_orientations: List[TensorType],
                          trajectory_features: TensorType['num_trajectories', 'trajectory_dim'],
                          trajectory_edge_features: List[TensorType],
                          directed: bool,) -> 'TrajectoryComplex':
        trajectory_edge_idxs, trajectory_edge_orientations = zip(*[edge_complex.oriented_edge_idxs_to_edge_idxs(edge_idxs, edge_orientations, directed)
                                for (edge_idxs, edge_orientations) in zip(trajectory_edge_idxs, trajectory_edge_orientations)])
        trajectory_edge_orientations = torch.cat(trajectory_edge_orientations)
        
        col = np.concatenate([edge_idxs.numpy() for edge_idxs in trajectory_edge_idxs])
        row = np.array(sum(([idx] * edge_idxs.size(0) for idx, edge_idxs in enumerate(trajectory_edge_idxs)), start=[]))
        edge_to_trajectory_adjacency = torch.stack((torch.from_numpy(col), torch.from_numpy(row))).long()
        if trajectory_edge_features is not None:
            trajectory_edge_features = torch.cat(trajectory_edge_features, dim=0)
            assert trajectory_edge_orientations.size(0) == trajectory_edge_features.size(0)
        
        # The commented block constructs adjacency between trajectories that share edges even if the orientation does not match, this should not be done (I think?)
        # num_original_edges = edge_complex.edge_features.size(0) // 2 if directed else edge_complex.edge_features.size(0)  # each edge was used to create edges for both directions
        # trajectory_indicator = sp.coo_matrix((np.ones_like(row), (row, col % num_original_edges)), shape=(len(trajectory_edge_idxs), num_original_edges,)) # T, E
        
        trajectory_indicator = sp.coo_matrix((np.ones_like(row), (row, col)), shape=(len(trajectory_edge_idxs), edge_complex.num_edges,)) # T, E
        trajectory_adjacency = (trajectory_indicator @ trajectory_indicator.T).tolil()
        trajectory_adjacency.setdiag(0)
        trajectory_adjacency = torch.from_numpy(np.array(trajectory_adjacency.nonzero())).long() # 2, T
        return TrajectoryComplex(edge_complex=edge_complex, 
                                 trajectory_features=trajectory_features, 
                                 edge_to_trajectory_adjacency=edge_to_trajectory_adjacency,
                                 trajectory_adjacency=trajectory_adjacency,
                                 edge_to_trajectory_orientation=trajectory_edge_orientations,
                                 edge_to_trajectory_adjacency_features=trajectory_edge_features)
    @staticmethod
    def collate_internal(batch: List['TrajectoryComplex']) -> Dict:
        """ Collates multiple trajectory complexes into a disjoint single one for batching.
        Returns kwargs for the constructor. """
        (node_features, edge_features, trajectory_features), (node_adjacency, edge_adjacency, trajectory_adjacency), \
            (node_adjacency_features, edge_adjacency_features, trajectory_adjacency_features), \
            (node_to_edge_adjacency, edge_to_trajectory_adjacency), (node_to_edge_orientation, edge_to_trajectory_orientation), \
            (node_to_edge_adjacency_features, edge_to_trajectory_adjacency_features), \
            (node_batch_idxs, edge_batch_idxs, trajectory_batch_idxs) = collate_complex(*zip(*[(
                [tc.edge_complex.node_features, tc.edge_complex.edge_features, tc.trajectory_features],
                [tc.edge_complex.node_adjacency, tc.edge_complex.edge_adjacency, tc.trajectory_adjacency],
                [tc.edge_complex.node_adjacency_features, tc.edge_complex.edge_adjacency_features, tc.trajectory_adjacency_features],
                [tc.edge_complex.node_to_edge_adjacency, tc.edge_to_trajectory_adjacency],
                [tc.edge_complex.node_to_edge_orientation, tc.edge_to_trajectory_orientation],
                [tc.edge_complex.node_to_edge_adjacency_features, tc.edge_to_trajectory_adjacency_features],
                [tc.edge_complex.node_batch_idxs, tc.edge_complex.edge_batch_idxs, tc.trajectory_batch_idxs],) for tc in batch]))
            
        return dict(
            trajectory_features=trajectory_features, trajectory_adjacency=trajectory_adjacency, trajectory_batch_idxs=trajectory_batch_idxs, 
            edge_to_trajectory_adjacency=edge_to_trajectory_adjacency,
            edge_to_trajectory_orientation=edge_to_trajectory_orientation,
            trajectory_adjacency_features=trajectory_adjacency_features,
            edge_to_trajectory_adjacency_features=edge_to_trajectory_adjacency_features,
            edge_complex=EdgeComplex(node_features=node_features,  edge_features=edge_features, 
                                     node_adjacency=node_adjacency,
                                     node_adjacency_edge_idxs=EdgeComplex.collate_node_adjacency_edge_idxs([b.edge_complex for b in batch]),
                                     edge_adjacency=edge_adjacency, node_to_edge_adjacency=node_to_edge_adjacency, 
                                     edge_adjacency_node_idxs=EdgeComplex.collate_edge_adjacency_node_idxs([b.edge_complex for b in batch]),
                                     node_batch_idxs=node_batch_idxs, edge_batch_idxs=edge_batch_idxs,
                                     node_to_edge_orientation=node_to_edge_orientation,
                                     batch_size=sum(b.batch_size for b in batch),
                                     graphs=sum((b.edge_complex.graphs for b in batch), start=[]),
                                     node_adjacency_sizes=sum((b.edge_complex.node_adjacency_sizes for b in batch), start=[]),
                                     node_adjacency_features=node_adjacency_features,
                                     edge_adjacency_features=edge_adjacency_features,
                                     node_to_edge_adjacency_features=node_to_edge_adjacency_features,
            ),
            attributes = sum((bi.attributes for bi in batch), start=[]))
    
    @staticmethod
    def collate(batch: List['TrajectoryComplex']) -> 'TrajectoryComplex':
        """ Collates multiple trajectory complexes into a disjoint single one for batching. """
        if len(batch) == 1:
            return batch[0]
        return TrajectoryComplex(**TrajectoryComplex.collate_internal(batch))
    
    @staticmethod
    def split_internal(batch: 'TrajectoryComplex') -> List[Dict]:
        """ Inverse of `collate`. Returns kwargs to the constructor of `TrajectoryComplex`. """
        features, adj_intra, features_adj_intra, adj_upper, adj_upper_orientation, features_adj_upper = split_complex(
            [batch.edge_complex.node_features, batch.edge_complex.edge_features, batch.trajectory_features],
            [batch.edge_complex.node_adjacency, batch.edge_complex.edge_adjacency, batch.trajectory_adjacency],
            [batch.edge_complex.node_adjacency_features, batch.edge_complex.edge_adjacency_features, batch.trajectory_adjacency_features],
            [batch.edge_complex.node_to_edge_adjacency, batch.edge_to_trajectory_adjacency],
            [batch.edge_complex.node_to_edge_orientation, batch.edge_to_trajectory_orientation],
            [batch.edge_complex.node_to_edge_adjacency_features, batch.edge_to_trajectory_adjacency_features],
            [batch.edge_complex.node_batch_idxs, batch.edge_complex.edge_batch_idxs, batch.trajectory_batch_idxs])
        
        return [dict(edge_complex=EdgeComplex(node_features=node_features, edge_features=edge_features,
                                                           node_adjacency=adj_nodes_intra, 
                                                           node_adjacency_edge_idxs=node_adj_edge_idxs,
                                                           node_to_edge_adjacency=adj_node_to_edge,
                                                           node_adjacency_features=features_adj_nodes_intra,
                                                           edge_adjacency_features=features_adj_edges_intra,
                                                           edge_adjacency_node_idxs=edge_adj_node_idxs,
                                                           node_to_edge_orientation=adj_node_to_edge_orientation,
                                                           node_to_edge_adjacency_features=features_adj_node_to_edge,
                                                           edge_adjacency=adj_edges_intra,
                                                           graphs=[graph],
                                                           node_adjacency_sizes=[node_adj_size]),
                                  trajectory_features=trajectory_features, edge_to_trajectory_adjacency=adj_edge_to_traj,
                                  edge_to_trajectory_adjacency_features=features_adj_edge_to_traj,
                                  trajectory_adjacency_features=features_adj_traj_intra,
                                  edge_to_trajectory_orientation=adj_edge_to_traj_orientation,
                                  trajectory_adjacency=adj_traj_intra, attributes=[a]) for (
                                      (node_features, edge_features, trajectory_features), 
                                      (adj_nodes_intra, adj_edges_intra, adj_traj_intra), 
                                      (features_adj_nodes_intra, features_adj_edges_intra, features_adj_traj_intra),
                                      (adj_node_to_edge, adj_edge_to_traj),
                                      (adj_node_to_edge_orientation, adj_edge_to_traj_orientation),
                                      (features_adj_node_to_edge, features_adj_edge_to_traj),
                                      graph,
                                      node_adj_size,
                                      a,
                                      node_adj_edge_idxs,
                                      edge_adj_node_idxs,) in 
                zip(features, adj_intra, features_adj_intra, adj_upper, adj_upper_orientation, features_adj_upper, batch.edge_complex.graphs, \
                    batch.edge_complex.node_adjacency_sizes, batch.attributes, EdgeComplex.split_node_adjacency_edge_idxs(batch.edge_complex),
                    EdgeComplex.split_edge_adjacency_node_idxs(batch.edge_complex))]
        
    @staticmethod
    def split(batch: 'TrajectoryComplex') -> List['TrajectoryComplex']:
        """ Inverse of `collate` """
        if batch.batch_size == 1:
            return [batch]
        return [TrajectoryComplex(**kwargs) for kwargs in TrajectoryComplex.split_internal(batch)]
    
    @property
    def batch_size(self) -> int:
        return self.edge_complex.batch_size
    
    def _map_to_all_tensors(self, _func):
        object.__setattr__(self, 'edge_complex', _func(self.edge_complex))
        if self.trajectory_features is not None:
            object.__setattr__(self, 'trajectory_features', _func(self.trajectory_features))
        object.__setattr__(self, 'trajectory_adjacency', _func(self.trajectory_adjacency))
        object.__setattr__(self, 'edge_to_trajectory_adjacency', _func(self.edge_to_trajectory_adjacency))
        object.__setattr__(self, 'trajectory_batch_idxs', _func(self.trajectory_batch_idxs))
        object.__setattr__(self, 'edge_to_trajectory_orientation', _func(self.edge_to_trajectory_orientation))
        if self.edge_to_trajectory_adjacency_features is not None:
            object.__setattr__(self, 'edge_to_trajectory_adjacency_features', _func(self.edge_to_trajectory_adjacency_features))
        if self.trajectory_adjacency_features is not None:
            object.__setattr__(self, 'trajectory_adjacency_features', _func(self.trajectory_adjacency_features))
    
    
    def pin_memory(self) -> 'TrajectoryComplex':
        self._map_to_all_tensors(lambda tensor: tensor.pin_memory())
        return self
    
    def to(self, device) -> 'TrajectoryComplex':
        self._map_to_all_tensors(lambda tensor: tensor.to(device))
        return self
