from typing import List, Tuple, Dict
from torchtyping import TensorType
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
import attrs
import numpy as np
from typeguard import typechecked
import scipy.sparse as sp

from utils.graph import plane_graph_compute_faces
from config import BaseDataConfig
from data.base import Graph, EdgeComplex, TrajectoryComplex, CellComplex

SHORTEST_PATH_DISTANCE_METRIC_PREFIX = 'sp_'

@attrs.define
class BaseDataset:
    """ Abstract base trajectory dataset. """
     
    node_features: TensorType['num_nodes', 'num_node_features'] | None
    node_feature_names: List[str]
    node_coordinates: TensorType['num_nodes', 'num_coordinates'] | None
    edge_idxs: TensorType['num_edges', 2]
    edge_features: TensorType['num_edges', 'num_edge_features'] | None
    edge_feature_names: List[str]
    trajectories: List[Dict]
    trajectory_feature_names: List[str] = attrs.field(factory=list)
    directed: bool=True
    observed_edges: List[int] | None = None
    
    # Cell complex attributes, all optional
    cell_complex_computed: bool = False
    discard_largest_face: bool = True # when computing faces, the largest face will be discarded (hopefully the exterior face)
    lower_edge_hodge_laplacian_idxs: TensorType[int, 2, 'num_lower_edge_hodge_laplacian'] | None = None
    lower_edge_hodge_laplacian_weights: TensorType[float, 'num_lower_edge_hodge_laplacian'] | None = None
    upper_edge_hodge_laplacian_idxs: TensorType[int, 2, 'num_upper_edge_hodge_laplacian'] | None = None
    upper_edge_hodge_laplacian_weights: TensorType[float, 'num_upper_edge_hodge_laplacian'] | None = None
    edge_to_cell_adjacency: TensorType[int, 2, 'num_edge_to_cell_adjacency'] | None = None
    edge_to_cell_adjacency_orientation: TensorType[int, 'num_edge_to_cell_adjacency'] | None = None
    cell_adjacency: TensorType[int, 2, 'num_cell_adjacency'] | None = None
    
    # meta-structures are built lazily
    _graph: Graph | None = None
    _faces: Tuple[List[List[int]], List[List[int]]] | None = None
    _edge_complex: EdgeComplex | None = None

    node_names: List[str] | None = None
    
    node_distance_metrics: Dict[str, TensorType['num_nodes', 'num_nodes']] | None = None
    
    name: str | None = None
        
    def __attrs_post_init__(self):
        if self.observed_edges is None:
            self._setup_observed_edges()
        if self.node_distance_metrics is None:
            self._setup_node_distance_metrics()
        
    def _setup_observed_edges(self):
        observed_edges = set()
        # Compute observed edges from all trajectories
        iterator = tqdm(self.trajectories, desc='Computing observed edges') if len(self.trajectories) > 1e5 else self.trajectories
        for trajectory in iterator:
            # TODO: directional support? probably not, as we don't want to accidentally remove a direction not observed
            observed_edges.update(trajectory['edge_idxs'].tolist())
        self.observed_edges = list(observed_edges)
        
    def _setup_node_distance_metrics(self):
        self.node_distance_metrics = {}
        self.add_shortest_path_node_distance_metric('hops', torch.ones(self.edge_idxs.size(0)))
        if self.node_coordinates is not None:
            self.add_node_distance_metric('euclidean', torch.cdist(self.node_coordinates, self.node_coordinates, p=2))
            self.add_shortest_path_node_distance_metric('euclidean', torch.norm(torch.diff(self.node_coordinates[self.edge_idxs], dim=1).squeeze(1), dim=1, p=2))
        
    @typechecked
    def add_node_distance_metric(self, key: str, distances: TensorType['num_nodes', 'num_nodes']):
        """ Adds a node distance metric. """
        assert not key in self.node_distance_metrics, f'Metric {key} already a shortest path distance metric'
        self.node_distance_metrics[key] = distances
        
    @typechecked
    def add_shortest_path_node_distance_metric(self, key: str, edge_weights: TensorType['num_edges']):
        """ Adds a shortest path-based metrics with given edge weights """ 
        edge_weights = edge_weights.repeat(2).detach().cpu().numpy()
        edge_idxs = torch.cat((self.edge_idxs, self.edge_idxs.flip(1)), dim=0)
        
        A_nodes = sp.coo_matrix(
            (edge_weights, edge_idxs.T.cpu().numpy()), 
            shape=(self.num_nodes, self.num_nodes)).tocsr()
        
        node_distance_matrix = torch.from_numpy(sp.csgraph.shortest_path(A_nodes, directed=False, return_predecessors=False)).float()
        self.add_node_distance_metric(f'{SHORTEST_PATH_DISTANCE_METRIC_PREFIX}{key}', node_distance_matrix)
        
    def reduce_graph_to_observed(self) -> 'ReducedDataset':
        """ Reduces the graph and (lazily) the trajectories to only observed edges and nodes. """
        observed_nodes = torch.tensor(list(set(torch.flatten(self.edge_idxs[self.observed_edges]).tolist())))
        node_reduction_map = -torch.ones(self.num_nodes, dtype=int)
        node_reduction_map[observed_nodes] = torch.arange(len(observed_nodes))
        edge_reduction_map = -torch.ones(self.num_edges_undirected, dtype=int)
        edge_reduction_map[self.observed_edges] = torch.arange(len(self.observed_edges))
        edge_idx_reduced = node_reduction_map[self.edge_idxs[self.observed_edges]]
        assert (edge_idx_reduced >= 0).all()
        
        return ReducedDataset(node_features=self.node_features[observed_nodes] if self.node_features is not None else None, 
                                node_feature_names=self.node_feature_names,
                                node_coordinates=self.node_coordinates[observed_nodes] if self.node_coordinates is not None else None,
                                edge_idxs=edge_idx_reduced,
                                edge_features=self.edge_features[self.observed_edges] if self.edge_features is not None else None,
                                edge_feature_names=self.edge_feature_names,
                                trajectories=self.trajectories,
                                directed=self.directed,
                                observed_edges=torch.arange(len(self.observed_edges), dtype=int),
                                node_reduction_map=node_reduction_map,
                                edge_reduction_map=edge_reduction_map
                                )
        
    @property
    def num_nodes(self) -> int: 
        if self.node_features is None:
            if self.edge_idxs.numel() == 0:
                return 0
            return self.edge_idxs.max().item() + 1
        else:
            return self.node_features.size(0)
    
    @property
    def num_node_features(self) -> int | None: 
        return self.node_features.size(1) if self.node_features is not None else 0
    
    @property
    def num_edge_features(self) -> int | None:
        return self.edge_features.size(1) if self.edge_features is not None else 0

    @property
    def num_trajectory_features(self) -> int | None:
        if len(self.trajectories) == 0:
            return 0
        trajectory = self.trajectories[0]
        return trajectory['features'].size(0) if trajectory['features'] is not None else 0
    
    @property
    def num_edges_undirected(self) -> int: return self.edge_idxs.size(0)
    
    @property
    def num_edges(self) -> int: return self.num_edges_undirected * (2 if self.directed else 1)
    
    @classmethod
    def from_config(cls, config: BaseDataConfig) -> 'BaseDataset':
        raise NotImplementedError

    @staticmethod
    def serialization_file_suffix() -> str: return '.pt'

    def serialize(self, path: str):
        torch.save({
            'node_features' : self.node_features,
            'node_feature_names' : self.node_feature_names,
            'node_coordinates' : self.node_coordinates,
            
            'edge_idxs' : self.edge_idxs,
            'edge_features' : self.edge_features,
            'edge_feature_names' : self.edge_feature_names,
            
            'trajectories' : self.trajectories,
            'trajectory_feature_names' : self.trajectory_feature_names,
            'directed' : self.directed,
            'observed_edges' : self.observed_edges,
            
            'discard_largest_face' : self.discard_largest_face,
            'node_names' : self.node_names,
            'node_distance_metrics' : self.node_distance_metrics,
            'name' : self.name,
        }, path)
    
    @classmethod
    def deserialize(cls, path: str) -> 'BaseDataset':
        storage = torch.load(path)
        return cls(**storage)
    
    def __len__(self) -> int: return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> dict: return self.trajectories[idx]
    
    def faces(self) -> Tuple[List[List[int]], List[List[int]]]:
        if self._faces is None:
            if self.node_coordinates is None:
                raise RuntimeError(f'Node coordinates of a plane graph need to be supplied to compute faces.')
            if self._graph is None:
                raise RuntimeError(f'Faces should be computed after the graph is computed, call `dataset.graph(...)` at least once.')
            faces = plane_graph_compute_faces(self.node_coordinates, self._graph.node_adjacency)
            if self.directed:
                # each direction has its own edge, therefore the orientation is 1
                faces, face_orientation = faces, [[1 for _ in face] for face in faces]
            else:
                faces, face_orientation = [[edge % self.num_edges_undirected for edge in face] for face in faces], \
                    [[1 if edge < self.num_edges_undirected else -1 for edge in face] for face in faces]
            # The ordering of faces is relevant to unittests, so we fix it by interpreting each edge-sequence as a decimal string
            face_idxs_sorted = sorted(range(len(faces)), key=lambda idx: faces[idx])
                    
            self._faces = [faces[idx] for idx in face_idxs_sorted], [face_orientation[idx] for idx in face_idxs_sorted]
        return self._faces
        
    def graph(self,) -> Graph:
        """ Creates the graph from the edge complex. 
        
        Returns:
        --------
        graph : Graph
            The graph instance.
        """
        if self._graph is None:
            self._graph = Graph(self.edge_idxs.T, self.directed, observed_edges=self.observed_edges, node_distance_metrics=self.node_distance_metrics)
        return self._graph
    
    def edge_complex(self) -> EdgeComplex:
        """ Gets the base edge simplicial complex representation of this dataset. 
        
        Clarification:
        --------------
        If `self.directed`, then each edge in the dataset graph will be represented by two edges, one
        for each direction. This means, each of these edges has its own features etc. The propagation
        will respect these orientations, i.e. an edge e1 and e2 will only be adjacent, when the endpoint
        of e1 is the startpoint of e2.
        If not `self.directed`, then each edge in the graph will be represented by only one edge, i.e.
        only one feature per edge etc. While the start/end points in the complex are arbitrary, the propagation
        will not regard orientation. That is, if any two points of e1 and e2 match, they will communicate.
        """
        if self._edge_complex is None:
            logging.info('Building dataset graph')
            graph = self.graph()
            edge_features = self.edge_features
            if edge_features is not None and graph.directed:
                edge_features = edge_features.repeat(2, 1)
            self._edge_complex = EdgeComplex.from_graph(self.graph(), 
                                                        self.node_features,
                                                        edge_features)
        return self._edge_complex
    
    def compute_cell_complex(self, recompute: bool=False):
        """ Precomputes the cell complex the dataset plane graph resides on. """
        if not recompute and self.cell_complex_computed:
            return
        
        edge_complex = self.edge_complex()
        graph = edge_complex.graphs[0]
        faces, face_edge_orientations = self.faces()
        
        # TODO: we want to remove the "exterior" face that extends to infinity
        # for now, we assume this is simply the largest face (as it spans around the entire graph)
        # this is hacky and potentially buggy, there probably is a nice geometric way to identfy this face
        face_sizes = [len(face) for face in faces]
        largest_face_idx = face_sizes.index(max(face_sizes))
        faces = [faces[idx] for idx in range(len(faces)) if idx != largest_face_idx]
        face_edge_orientations = [face_edge_orientations[idx] for idx in range(len(face_edge_orientations)) if idx != largest_face_idx]
        
        # Compute the upper and lower hodge laplacian components for edges
        # lower hodge laplacian constructed from signed node->edge indicator
        node_to_edge_signed_indicator = graph.node_to_edge_startpoint_indicator - graph.node_to_edge_endpoint_indicator.T
        lower_edge_hodge_laplacian = node_to_edge_signed_indicator.T @ node_to_edge_signed_indicator
        lower_edge_hodge_laplacian_idxs = lower_edge_hodge_laplacian.nonzero()
        lower_edge_hodge_laplacian_weights = lower_edge_hodge_laplacian[lower_edge_hodge_laplacian_idxs]
        self.lower_edge_hodge_laplacian_idxs = torch.from_numpy(np.array(lower_edge_hodge_laplacian_idxs, dtype=int)).long()
        self.lower_edge_hodge_laplacian_weights = torch.from_numpy(np.array(lower_edge_hodge_laplacian_weights, dtype=float)).float()
        
        # upper hodge laplacian constructed from signed edge->face indicator
        edge_to_face_idxs, edge_to_face_orientations = [], []
        for face_idx, (face, face_edge_orientation) in enumerate(zip(faces, face_edge_orientations)):
            for edge_idx, orientation in zip(face, face_edge_orientation):
                edge_to_face_idxs.append((edge_idx, face_idx))
                edge_to_face_orientations.append(orientation)
                if graph.directed:
                    # each face e1, .. ek also is represented by -e1, ... -ek
                    # but only counted once by the `faces` algorithm
                    # we manually make indicendes of -1 to the face for inverted edges
                    assert orientation == 1
                    edge_to_face_idxs.append(((edge_idx + graph.num_edges // 2) % graph.num_edges, face_idx))
                    edge_to_face_orientations.append(-1)
                
        edge_to_face_idxs = np.array(edge_to_face_idxs)
        edge_to_face_orientations = np.array(edge_to_face_orientations)
        
        edge_to_face_signed_indicator = sp.coo_matrix((edge_to_face_orientations, edge_to_face_idxs.T), dtype=int,)
        upper_edge_hodge_laplacian = edge_to_face_signed_indicator @ edge_to_face_signed_indicator.T
        upper_edge_hodge_laplacian_idxs = upper_edge_hodge_laplacian.nonzero()
        upper_edge_hodge_laplacian_weights = upper_edge_hodge_laplacian[upper_edge_hodge_laplacian_idxs]
        self.upper_edge_hodge_laplacian_idxs = torch.from_numpy(np.array(upper_edge_hodge_laplacian_idxs, dtype=int))
        self.upper_edge_hodge_laplacian_weights = torch.from_numpy(np.array(upper_edge_hodge_laplacian_weights)).float()
        
        self.edge_to_cell_adjacency, self.edge_to_cell_adjacency_orientation, self.cell_adjacency = \
            CellComplex.compute_cell_adjacency(edge_complex, faces, face_edge_orientations)
        self.cell_complex_computed = True
    
    @typechecked
    def reduced_edge_complex(self, node_idxs: TensorType['num_nodes_reduced']) -> Tuple[EdgeComplex, TensorType['num_nodes'], TensorType['num_edges'],
                                                                                        TensorType['num_nodes_reduced'], TensorType['num_edges_reduced']]:
        edge_complex = self.edge_complex()
        assert edge_complex.batch_size == 1, f'Should reduce complex before collating, i.e. batch size should be 1'
        graph, node_reduction_map, edge_reduction_map, node_reduction_map_inverse, edge_reduction_map_inverse = edge_complex.graphs[0].reduce(node_idxs)
        edge_features = self.edge_features
        if edge_features is not None and graph.directed:
            edge_features = edge_features.repeat(2, 1)
        
        return EdgeComplex.from_graph(
            graph,
            self.node_features[node_reduction_map_inverse] if self.node_features is not None else None,
            edge_features[edge_reduction_map_inverse] if self.edge_features is not None else None,
        ), node_reduction_map, edge_reduction_map, node_reduction_map_inverse, edge_reduction_map_inverse
        

@attrs.define
class ReducedDataset(BaseDataset):
    """ Dataset that has a reduced set of nodes and edges. """
    node_reduction_map: TensorType['num_nodes_original'] = None
    edge_reduction_map: TensorType['num_edges_original'] = None
    
    def __getitem__(self, idx: int) -> dict:
        trajectory = super().__getitem__(idx).copy()
        trajectory['edge_idxs'] = self.edge_reduction_map[trajectory['edge_idxs']]
        assert (trajectory['edge_idxs'] >= 0).all()
        return trajectory

class TrajectoryComplexDataset(Dataset):
    """ A torch dataset where each instance is a trajectory complex with one trajectory. """
    
    def __init__(self, base_dataset: BaseDataset, in_memory: bool = True):
        
        self.in_memory = in_memory
        self.base_dataset = base_dataset
        self.edge_complex = base_dataset.edge_complex()
        if self.in_memory:
            self._trajectory_complexes = []
            for trajectory in base_dataset.trajectories:
                self._trajectory_complexes.append(
                    self._build_item(trajectory)
                )
        else:
            raise NotImplemented
        
    def _build_item(self, trajectory) -> TrajectoryComplex:
        """ Computes one trajectory complex of the dataset. """
        return TrajectoryComplex.from_edge_complex(self.edge_complex,
                        [trajectory.edge_idxs], [trajectory.edge_orientation], trajectory.features.reshape((1, -1)),)
                
    def __len__(self) -> int:
        if self.in_memory:
            return len(self._trajectory_complexes)
        else:
            raise NotImplementedError
    
    def __getitem__(self, idx: int) -> TrajectoryComplex:
        if self.in_memory:
            return self._trajectory_complexes[idx]
        else:
            raise NotImplementedError
        
        