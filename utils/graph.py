import numpy as np
from numpy.typing import NDArray
from typing import Dict, Set, Tuple, List
from collections import defaultdict
from itertools import product
from torchtyping import TensorType
from typeguard import typechecked
import torch
import torch_scatter
    
from utils.utils import compute_2d_angle_to_x_axis

def graph_to_adjacency_list(edges: NDArray | TensorType['num_edges', 2]) -> Dict[int, Set[int]]:
    """Creates an adjacency list for each node

    Parameters
    ----------
    edges : NDArray | TensorType[&#39;num_edges&#39;, 2]
        _description_

    Returns
    -------
    Dict[Set[int]]
        _description_
    """
    adjacency_list = defaultdict(set)
    for u, v in edges:
        adjacency_list[u].add(v)
        adjacency_list[v].add(u)
    return adjacency_list

def graph_all_paths(edges: NDArray, length: int) -> List[Set[Tuple[int]]]:
    """Finds all paths of a given length in a graph, grouped by their start node.

    Parameters
    ----------
    edges : NDArray, shape [m, 2]
        edge indices, can be directed or undirected, but the graph is interpreted as undirected
    length : int
        the length of all paths

    Returns
    -------
    List[Set[Tuple[int]]]
        for each node, a set of paths (length-tuples)
    """
    adjacency_list = graph_to_adjacency_list(edges)
    num_nodes = edges.max() + 1
    # Dynamic programming
    paths = [[set() for _ in range(num_nodes)] for _ in range(length + 1)]
    for u in range(num_nodes):
        paths[0][u].add((u,))
    for k in range(1, length + 1):
        for u in range(num_nodes):
            paths[k][u].update((u,) + other for v in adjacency_list[u] for other in paths[k - 1][v] )
    return paths[length]
        
def plane_graph_compute_faces(node_coordinates: TensorType['num_nodes', 2],
                              node_adjacency: TensorType[2, 'num_edges_directed'],
                              ) -> List[List[int]]:
    """ Computes the faces of a plane graph. 
    
    Parameters:
    -----------
    node_coordinates : TensorType['num_nodes', 2]
        Spatial coordinates of the nodes in the plane graph.
    node_adjacency : TensorType[2, 'num_edges_directed']
        Adjacency of nodes (i.e. edges). This should contain both pairs (u, v)
        and (v, u). Also, it should be ordered like: [(a, b), (c, d), ..., (b, a), (d, c), ...]
        Will be asserted.
    directed : bool
        Whether the output refers to a directed graph or an undirected graph.
    
    Returns:
    --------
    faces : List[List[int]]
        A list of faces, identifying indices in `node_adjacency` as constituents of each face.
    """

    edge_endpoint_coordinates = node_coordinates[node_adjacency]
    # we treat each edge with a direction, m is the directed number of edges
    m = node_adjacency.size(1)
    assert (node_adjacency[:, :m // 2] == node_adjacency[:, m // 2:].flip(0)).all()

    # for each node, store all outgoing edges and order them according to the angle on the x axis
    edge_vectors = torch.diff(edge_endpoint_coordinates, dim=0).squeeze(0) # E,2
    edge_angles = compute_2d_angle_to_x_axis(edge_vectors)
    edges = node_adjacency.T.tolist()
    num_nodes = node_coordinates.size(0)
    
    node_to_outgoing_edge_list = [[] for _ in range(num_nodes)]
    for edge_idx, (u, v) in enumerate(edges):
        node_to_outgoing_edge_list[u].append(edge_idx)
    node_to_outgoing_edge_list = [sorted(edge_list, key=lambda edge_idx: edge_angles[edge_idx]) for edge_list in node_to_outgoing_edge_list]

    faces = []

    # the main algorithm: start at any edge, at its endpoint pick the next edge in ccw direction and iterate until cycles close
    # each cycle will be face of the plane graph
    open_edges = set(range(edge_endpoint_coordinates.size(1)))
    while len(open_edges) > 0: # in O(m) compute all faces
        start_edge = next(iter(open_edges)) # random edge
        face = []
        current_edge = start_edge

        while True:
            face.append(current_edge)
            open_edges.remove(current_edge)

            # compute the next edge of the endpoint node in ccw direction
            # since `node_to_outgoing_edge_list` orders outgoing edges, we need to translate
            # the incoming edge `current_edge` to its corresponding outgoing edge, which is found in 
            # `node_to_outgoing_edge_list` of its endpoint
            incoming_edge = (m // 2) + current_edge if current_edge < (m // 2) else current_edge - (m // 2)
            u, v = edges[current_edge]
            adj = node_to_outgoing_edge_list[v]
            idx_current = adj.index(incoming_edge)
            current_edge = adj[(idx_current + 1) % len(adj)]

            if current_edge == start_edge:
                faces.append(face)
                break
    return faces
     
def add_self_loops_to_degree_zero_nodes(node_adjacency: TensorType[2, 'num_adj'], num_nodes: int | None = None, fill_value: float = 1.0,
                                        weights: TensorType['num_adj', -1] | None = None) -> Tuple[TensorType[2, 'num_adj_new'], TensorType['num_adj_new', -1] | None]:
    """ Adds self loops to a graph but only to nodes with degree zero.

    Parameters:
    -----------
    node_adjacency : TensorType[2, 'num_adj']
        The adjacency to add self-loops to. Interpreted as directed graph, i.e. node degrees are computed from the first index.
    num_nodes : int, optional
        How many nodes there are in the graph
    fill_value : float, optional, 1.0 by default
        With which weight to fill
    weights : TensorType['num_adj', ...], optional
        The weights of the adjacency matrix.

    Returns
    -------
    node_adjacency : TensorType[2, 'num_adj_new']
        The new adjacency matrix
    weights : TensorType['num_adj_new', ...], optional
        The new weights
    """
    idx_src, idx_dst = node_adjacency
    out_degree = torch_scatter.scatter_add(torch.ones_like(idx_src, dtype=int), idx_src, dim=0, dim_size=num_nodes)
    node_idxs = (out_degree == 0).nonzero()[:, 0]
    node_adjacency_new = torch.stack((torch.cat((idx_src, node_idxs)), torch.cat((idx_dst, node_idxs))))
    if weights is not None:
        size = (node_idxs.size(0), *list(weights.size())[1:])
        weights = torch.cat((weights, torch.full(size, fill_value, device=weights.device)))
    return node_adjacency_new, weights

@typechecked
def graph_diffusion(signal: TensorType, adjacency: TensorType[2, 'num_adj'], weights: TensorType | None, 
                    num_steps: int, return_all_steps: bool = False) -> List[TensorType] | TensorType:
    """ Diffuses a signal on a graph for `num_steps` steps. 
    
    Parameters:
    -----------
    signal : TensorType['num_nodes', 'd']
        The signal to diffuse
    adjacency : TensorType[2, 'num_adj']
        The adjacency matrix of the graph diffusion
    weights : TensorType['num_adj', -1] | None
        The weights of the graph diffusion
    num_steps : int
        How many diffusion steps to do
    return_all_steps : bool
        If True, a List of signals is returned that represents each step.
        Otherwise, only the final signal is returned.
    
    Returns:
    --------
    List[TensorType['num_nodes', 'd']] | TensorType['num_nodes', 'd']
        Either signals after each step if `return_all_steps` is `True`, otherwise the final signal.
    """
    num_nodes = signal.size(0)
    results = []
    idx_src, idx_target = adjacency
    if weights is not None:
        assert adjacency.size(1) == weights.size(0)
    for _ in range(num_steps):
        message = signal[idx_src]
        if weights is not None:
            message *= weights
        signal = torch_scatter.scatter_add(message, idx_target, dim=0, dim_size=num_nodes)
        if return_all_steps:
            results.append(signal)
    
    if return_all_steps:
        return results
    else:
        return signal
        
    
    
    
    
    