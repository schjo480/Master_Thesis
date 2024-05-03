from copy import deepcopy, copy
import itertools
import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Iterable
from torchtyping import TensorType
from typeguard import typechecked
from collections import defaultdict
from bidict import bidict
import logging

import torch
from trajectory_gnn.utils.utils import FrozenMetaclass

class Graph(metaclass=FrozenMetaclass):
    """ Data structure for fastly navigating successor queries. 
    
    Parameters:
    -----------
    edge_idxs : TensorType[2, 'num_edges_original']
        Edge indices to base the graph on. Edges should be undirected and contain no duplicates.
        The ordering of star- and endpoint is arbitrary.
    directed : bool
        If directed, each undirected edge is translated into two directional edges.
        If not directed, each edge is treated as an undirected object.
    num_nodes : int, optional
        If given, over how many nodes the graph is constructed. If not given, inferred from the edge indices.
    node_distance_metrics : Dict[str, TensorType['num_nodes', 'num_nodes']], optional
        Pre-computed distance metrics for nodes.
        
    observed_edges : Iterable[int]
        Edges that were at least observed once (i.e. part of one trajectory)
    """
    
    def __init__(self, edge_idxs: TensorType[2, 'num_edges_original'], directed: bool, num_nodes: int | None = None, 
                 num_edges: int | None = None,
                 node_distance_metrics: Dict[str, TensorType['num_nodes', 'num_nodes']] | None = None,
                 observed_edges: Iterable[int] | None = None,
                 ):
        self.directed = directed
        self.node_adjacency_lists = defaultdict(list) # node_idx -> node_idx, edge_idx, edge_orientation
        self.edge_endpoints = bidict()
        self.num_nodes = num_nodes
        self.num_edges = num_edges or (2 * edge_idxs.size(1) if self.directed else edge_idxs.size(1))
        if observed_edges is None:
            observed_edges = range(edge_idxs.size(1))
        
        # Setup adjacency structres
        if self.num_nodes is None:
            self.num_nodes = edge_idxs.max().item() + 1 if edge_idxs.numel() > 0 else 0
        if self.directed:
            self._setup_directed(edge_idxs, self.num_nodes, observed_edges)
        else:
            self._setup_undirected(edge_idxs, self.num_nodes, observed_edges)
        node_distance_metrics = node_distance_metrics or {}
        self.node_distance_metrics = node_distance_metrics
          
    def _setup_directed(self, edge_idxs: TensorType[2, 'num_edges_original'], num_nodes: int,
                        observed_edges: Iterable[int]):
        """ Each possible orientation is represented by its own edge. """
        
        self.node_adjacency = torch.hstack((edge_idxs, edge_idxs.flip(0)))
        self.node_adjacency_edge_idxs = torch.arange(2 * edge_idxs.size(1))
        self.node_adjacency_edge_orientation = torch.zeros(2 * edge_idxs.size(1))
        
        node_adjacency_in_lists = defaultdict(list)
        for edge_idx, (u, v) in enumerate(zip(*self.node_adjacency.tolist())):
            self.node_adjacency_lists[u].append((v, (edge_idx, 0)))
            node_adjacency_in_lists[v].append((u, (edge_idx, 1)))
            self.edge_endpoints[edge_idx] = (u, v)

        edge_adjacency_idx_to_node_adjacency_idx = []
        edge_adjacency, edge_adjacency_node_idxs = [], []
        
        
        for v, nbs in self.node_adjacency_lists.items():
            new_pairs = [(e1, e2) for (e1, e2) in itertools.product(
                    (e for (_, (e, _)) in nbs),
                    (e for (_, (e, _)) in node_adjacency_in_lists[v]),
                ) if e1 != e2]
            # connect (u, v) <-> (v, w) via v
            edge_adjacency += new_pairs
            edge_adjacency_node_idxs += [v] * len(new_pairs)
            edge_adjacency_idx_to_node_adjacency_idx += [e1 for (e1, e2) in new_pairs]
            # symmetrize: connect (v, u) <-> (w, v) via v
            edge_adjacency += [(e2, e1) for (e1, e2) in new_pairs]
            edge_adjacency_node_idxs += [v] * len(new_pairs)
            edge_adjacency_idx_to_node_adjacency_idx += [e1 for (e1, e2) in new_pairs]
            
        self.edge_adjacency = torch.tensor(edge_adjacency, dtype=int).view(-1, 2).T
        self.edge_adjacency_node_idxs = torch.tensor(edge_adjacency_node_idxs, dtype=int)
        self.edge_adjacency_idx_to_node_adjacency_idx = torch.tensor(edge_adjacency_idx_to_node_adjacency_idx, dtype=int)
        
        self.node_to_edge_adjacency = torch.stack((self.node_adjacency.flatten(), torch.arange(self.node_adjacency.size(1)).repeat(2)))
        self.node_to_edge_orientation = torch.cat((torch.ones(self.node_adjacency.size(1)), -torch.ones(self.node_adjacency.size(1))))
        # node_to_edge features are one-hot indicators of start- and endpoints
        self.node_to_edge_adjacency_features = torch.zeros(self.node_to_edge_orientation.size(0), 2)
        self.node_to_edge_adjacency_features[: self.node_adjacency.size(1), 0] = 1
        self.node_to_edge_adjacency_features[self.node_adjacency.size(1) :, 1] = 1

        # Setup start- and endpoint indicator matrix
        self.node_to_edge_startpoint_indicator = sp.csc_matrix((np.ones(self.node_adjacency.size(1)), (self.node_adjacency[0].numpy(), range(self.node_adjacency.size(1)),)),
                                                shape=(self.num_nodes, self.num_edges)) # V x E
        self.node_to_edge_endpoint_indicator = sp.csr_matrix((np.ones(self.node_adjacency.size(1)), (range(self.node_adjacency.size(1)), self.node_adjacency[1].numpy())), 
                                            shape=(self.num_edges, self.num_nodes)) # E x V
        
        # which edges were observed
        self.observed_edges = torch.tensor(list(set(observed_edges)) + list(set(e + edge_idxs.size(1) for e in observed_edges)), dtype=int)
        self.observed_nodes = torch.tensor(list(set(torch.flatten(self.node_adjacency[:, self.observed_edges]).tolist())), dtype=int)
        
    def _setup_undirected(self, edge_idxs: TensorType[2, 'num_edges_original'], num_nodes: int,
                          observed_edges: Iterable[int]):
        """ Each edge represents both directions. """
        
        self.node_adjacency = torch.hstack((edge_idxs, edge_idxs.flip(0)))
        self.node_adjacency_edge_idxs = torch.arange(edge_idxs.size(1)).repeat(2)
        self.node_adjacency_edge_orientation = torch.cat([torch.zeros(edge_idxs.size(1)), torch.ones(edge_idxs.size(1))])
        
        for edge_idx, (u, v) in enumerate(zip(*edge_idxs.tolist())): # arbitrary orientation, the graph's are undirected
            self.node_adjacency_lists[u].append((v, (edge_idx, 0)))
            self.node_adjacency_lists[v].append((u, (edge_idx, 1)))
            self.edge_endpoints[edge_idx] = (u, v)
        
        # Setup start- and endpoint indicator matrix. Note that this orientation is arbitrary
        self.node_to_edge_startpoint_indicator = sp.csc_matrix((np.ones(self.num_edges), (self.node_adjacency[0, :self.num_edges].numpy(), range(self.num_edges),)),
                                                shape=(self.num_nodes, self.num_edges)) # V x E
        self.node_to_edge_endpoint_indicator = sp.csr_matrix((np.ones(self.num_edges), (range(self.num_edges), self.node_adjacency[1, :self.num_edges].numpy())), 
                                            shape=(self.num_edges, self.num_nodes)) # E x V

        edge_adjacency = []
        edge_adjacency_node_idxs = []
        edge_adjacency_idx_to_node_adjacency_idx = []
        for v, nbs in self.node_adjacency_lists.items():
            
            # new_pairs = [(e1, e2) for (e1, e2) in itertools.product((e for _, (e, _) in nbs), repeat=2) if e1 != e2]
            # edge_adjacency += new_pairs
            new_pairs = [(e1, o1, e2, o2) for ((e1, o1), (e2, o2)) in itertools.product((eo for _, eo in nbs), repeat=2) if e1 != e2]
            edge_adjacency += [(e1, e2) for (e1, o1, e2, o2) in new_pairs]
            edge_adjacency_node_idxs += [v] * len(new_pairs)
            edge_adjacency_idx_to_node_adjacency_idx += [e2 + o2 * self.num_edges for (e1, o1, e2, o2) in new_pairs]
                
                
        self.edge_adjacency = torch.tensor(edge_adjacency, dtype=int).view(-1, 2).T
        self.edge_adjacency_node_idxs = torch.tensor(edge_adjacency_node_idxs, dtype=int)
        self.edge_adjacency_idx_to_node_adjacency_idx = torch.tensor(edge_adjacency_idx_to_node_adjacency_idx, dtype=int)
        
        self.node_to_edge_adjacency = torch.stack((edge_idxs.flatten(), torch.arange(edge_idxs.size(1)).repeat(2)))
        self.node_to_edge_orientation = torch.cat((torch.ones(edge_idxs.size(1)), -torch.ones(edge_idxs.size(1)))) # arbitrary
        # node_to_edge features are one-hot indicators of start- and endpoints
        self.node_to_edge_adjacency_features = torch.ones(self.node_to_edge_orientation.size(0), 2)
        
        # which edges were observed
        self.observed_edges = torch.tensor(list(set(observed_edges)), dtype=int)
        self.observed_nodes = torch.tensor(list(set(torch.flatten(edge_idxs[:, self.observed_edges]).tolist())), dtype=int)
    
    def successors_by_edge(self, edge_idx : int, edge_orientation : int) -> List[Tuple[int, Tuple[int, int]]]:
        """Returns the sucessors (i.e. neighbours) of an edge.

        Parameters
        ----------
        edge_idx : int
            The edge to return neighbours of
        edge_orientation : int
            The orientation of the edge to return neighbours of

        Returns
        -------
        List[Tuple[int, Tuple[int, int]]]
            A list of (successor_node, (sucessor_edge, successor_edge_orientation))
        """
        start, end = self.edge_endpoints[edge_idx]
        if edge_orientation == 0:
            return self.successors_by_node(end)
        else:
            return self.successors_by_node(start)
    
    def oriented_edge_endpoints(self, edge_idx: int, orientation: int) -> Tuple[int, int]:
        """Gets the start- and endpoint of an (oriented edge)

        Parameters
        ----------
        edge_idx : int
            the edge index
        orientation : int
            the edge orientation

        Returns
        -------
        Tuple[int, int]
            start and endpoint
        """
        start, end = self.edge_endpoints[edge_idx]
        if orientation == 0:
            return start, end
        else:
            return end, start
    
    def successors_by_node(self, node_idx: int) -> List[Tuple[int, Tuple[int, int]]]:
        """Returns the sucessors (i.e. neighbours) of a node.

        Parameters
        ----------
        node_idx : int
            The node to return successors (i.e. neighbours) of

        Returns
        -------
        List[Tuple[int, Tuple[int, int]]]
            A list of (successor_node, (sucessor_edge, successor_edge_orientation))
        """
        return self.node_adjacency_lists[node_idx]
    
    @typechecked
    def reduce(self, node_idxs: TensorType['num_new_nodes'],) \
        -> Tuple['Graph', TensorType['num_nodes_old'], TensorType['num_edges_old'], TensorType['num_new_nodes'], TensorType['num_new_edges']]:
        """ Reduces the graph to a subset of nodes. 
        
        Parameters:
        -----------
        node_idxs : TensorType['num_new_nodes']
            Which nodes to reduce the graph to
        
        Returns:
        --------
        Graph
            The reduced graph instance
        node_reduction_map : TensorType['num_nodes_old']
            Mapping from old node index to new node index (-1 if the node is not present anymore)
        edge_reduction_map : TensorType['num_edges_old']
            Mapping from old edge index to new edge index (-1 if the edge is not present anymore)
        node_reduction_map_inverse : TensorType['num_new_nodes']
            Inverse mapping of `node_reduction_map`
        edge_reduction_map_inverse : TensorType['num_new_edges']
            Inverse mapping of `edge_reduction_map`
        """
        node_set = set(node_idxs.tolist())
        node_mask = torch.zeros(self.num_nodes, dtype=bool)
        node_mask[node_idxs] = True
        edge_mask = torch.zeros(self.num_edges, dtype=bool)
        edge_idxs = torch.tensor([edge_idx for edge_idx, (u, v) in self.edge_endpoints.items() if u in node_set and v in node_set])
        edge_set = set(edge_idxs.tolist())
        edge_mask[edge_idxs] = True
            
        # get maps old index to new index
        node_reduction_map = torch.full_like(node_mask, -1, dtype=int)
        node_reduction_map[node_mask] = torch.arange(len(node_set))
        node_reduction_map_inverse = torch.where(node_mask)[0] # this relies on the fact that where goes sequentially through the tensor (?)
        edge_reduction_map = torch.full_like(edge_mask, -1, dtype=int)
        edge_reduction_map[edge_mask] = torch.arange(len(edge_set))
        edge_reduction_map_inverse = torch.where(edge_mask)[0] # this relies on the fact that where goes sequentially through the tensor (?)
        
        node_reduction_map_dict = {old : new for old, new in enumerate(node_reduction_map.tolist())}
        edge_reduction_map_dict = {old : new for old, new in enumerate(edge_reduction_map.tolist())}
        
        reduced: 'Graph' = copy(self)
        with reduced.unfrozen():
            reduced.num_nodes = len(node_set)
            reduced.num_edges = len(edge_set)
            reduced.node_adjacency_lists = defaultdict(list) | {
                node_reduction_map_dict[u] : [
                    (node_reduction_map_dict[v], (edge_reduction_map_dict[edge_idx], orientation)) \
                    for v, (edge_idx, orientation) in adjacency_list
                    if v in node_set and edge_idx in edge_set
                ] for u, adjacency_list  in reduced.node_adjacency_lists.items()
                if u in node_set
            }
            
            reduced.edge_endpoints = bidict({
                edge_reduction_map_dict[e] : (node_reduction_map_dict[u], node_reduction_map_dict[v]) for
                e, (u, v) in reduced.edge_endpoints.items()
                if e in edge_set and u in node_set and v in node_set
            })
            
            node_adjacency_mask = node_mask[reduced.node_adjacency].all(0)
            
            node_adjacency_reduction_map = torch.full_like(node_adjacency_mask, -1, dtype=int)
            node_adjacency_reduction_map[node_adjacency_mask] = torch.arange(node_adjacency_mask.sum())
            reduced.node_adjacency = node_reduction_map[reduced.node_adjacency[:, node_adjacency_mask]]
            reduced.node_adjacency_edge_idxs = edge_reduction_map[reduced.node_adjacency_edge_idxs[node_adjacency_mask]]
            reduced.node_adjacency_edge_orientation = reduced.node_adjacency_edge_orientation[node_adjacency_mask]
            
            edge_adjacency_mask = edge_mask[reduced.edge_adjacency].all(0)
            reduced.edge_adjacency = edge_reduction_map[reduced.edge_adjacency[:, edge_adjacency_mask]]
            reduced.edge_adjacency_node_idxs = node_reduction_map[reduced.edge_adjacency_node_idxs[edge_adjacency_mask]]
            reduced.edge_adjacency_idx_to_node_adjacency_idx = node_adjacency_reduction_map[reduced.edge_adjacency_idx_to_node_adjacency_idx[edge_adjacency_mask]]
            
            # node_to_edge_adjacency
            node_to_edge_adjacency_node_mask = torch.stack((node_mask[reduced.node_to_edge_adjacency[0]],
                                                            edge_mask[reduced.node_to_edge_adjacency[1]])).all(0)
            reduced.node_to_edge_adjacency = reduced.node_to_edge_adjacency[:, node_to_edge_adjacency_node_mask]
            reduced.node_to_edge_adjacency = torch.stack((node_reduction_map[reduced.node_to_edge_adjacency[0]],
                                                          edge_reduction_map[reduced.node_to_edge_adjacency[1]]))
            reduced.node_to_edge_orientation = reduced.node_to_edge_orientation[node_to_edge_adjacency_node_mask]
            reduced.node_to_edge_adjacency_features = reduced.node_to_edge_adjacency_features[node_to_edge_adjacency_node_mask]
            
            # rebuilding indicator matrices is (probably) the fastest
            reduced.node_to_edge_startpoint_indicator = sp.csc_matrix((np.ones(reduced.num_edges), (reduced.node_adjacency[0, :reduced.num_edges].numpy(), range(reduced.num_edges),)),
                                                    shape=(reduced.num_nodes, reduced.num_edges)) # V x E
            reduced.node_to_edge_endpoint_indicator = sp.csr_matrix((np.ones(reduced.num_edges), (range(reduced.num_edges), reduced.node_adjacency[1, :reduced.num_edges].numpy())), 
                                                shape=(reduced.num_edges, reduced.num_nodes)) # E x V
            
            # which nodes are observed
            reduced.observed_nodes = node_reduction_map[reduced.observed_nodes[node_mask[reduced.observed_nodes]]]
            reduced.observed_edges = edge_reduction_map[reduced.observed_edges[edge_mask[reduced.observed_edges]]]
            
            # distance metrics
            reduced.node_distance_metrics = {
                name : distance_matrix[node_mask][:, node_mask]
                for name, distance_matrix in reduced.node_distance_metrics.items()
            }
            
        return reduced, node_reduction_map, edge_reduction_map, node_reduction_map_inverse, edge_reduction_map_inverse
    
        
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, Graph):
            return self.directed == __o.directed and self.node_adjacency_lists == __o.node_adjacency_lists and \
                self.edge_endpoints == __o.edge_endpoints and self.num_nodes == __o.num_nodes and self.num_edges == __o.num_edges and \
                (self.observed_edges == __o.observed_edges).all() and self.node_distance_metrics.keys() == __o.node_distance_metrics.keys() and \
                all((self.node_distance_metrics[k] == __o.node_distance_metrics[k]).all() for k in self.node_distance_metrics) and \
                (self.node_adjacency == __o.node_adjacency).all() and (self.node_adjacency_edge_idxs == __o.node_adjacency_edge_idxs).all() and \
                (self.node_adjacency_edge_orientation == __o.node_adjacency_edge_orientation).all() and \
                (self.edge_adjacency == __o.edge_adjacency).all() and (self.edge_adjacency_node_idxs == __o.edge_adjacency_node_idxs).all() and \
                (self.edge_adjacency_idx_to_node_adjacency_idx == __o.edge_adjacency_idx_to_node_adjacency_idx).all() and \
                (self.node_to_edge_adjacency == __o.node_to_edge_adjacency).all() and (self.node_to_edge_orientation == __o.node_to_edge_orientation).all() and \
                (self.node_to_edge_adjacency_features == __o.node_to_edge_adjacency_features).all() and \
                not (self.node_to_edge_startpoint_indicator != __o.node_to_edge_startpoint_indicator).toarray().any() and \
                not (self.node_to_edge_endpoint_indicator != __o.node_to_edge_endpoint_indicator).toarray().any() and \
                (self.observed_nodes == __o.observed_nodes).all()
        else:
            return False