from functools import partial
import torch

from torch.utils.data import Dataset
import numpy as np
import warnings
from tqdm import tqdm
import logging

from trajectory_gnn.data.base import *
from trajectory_gnn.data.base import TrajectoryComplex, BaseDataset
from trajectory_gnn.config import DataConfig
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingBatch

@typechecked
def reduce_edge_complex_to_k_hop_neighbourhood(k: int, base_dataset: BaseDataset, edge_idxs: TensorType['num_edges']) -> Tuple[EdgeComplex, TensorType['num_edges']]:
    """ Reduces the edge complex to the k-hop neighbourhood of an edge set. Use partial application of `k` to get a reduction function
    that can be used by `TrajectoryForecastingDataset`
    
    Parameters:
    -----------
    k : int
        The k hops.
    base_dataset : BaseDataset
        The base dataset containing the graph to reduce
    edge_idxs : TensorType['num_edges']
        The edge set.

    Returns
    -------
    EdgeComplex
        The reduced edge complex
    TensorType['num_edges']
        The edge set with indices on the reduced complex.
    """
    edge_complex = base_dataset.edge_complex()
    assert edge_complex.batch_size == 1
    node_idxs = torch.unique(edge_complex.node_adjacency[:, edge_complex.edge_idxs_to_node_adjacency_idxs(edge_idxs)].flatten())
    distance = edge_complex.graphs[0].node_distance_metrics['sp_hops'][node_idxs]
    node_idxs_reduced = torch.where((distance <= k).any(0))[0]
    edge_complex_reduced, node_reduction_map, edge_reduction_map, node_reduction_map_inverse, edge_reduction_map_inverse = \
        base_dataset.reduced_edge_complex(node_idxs_reduced)
    edge_idxs_reduced = edge_reduction_map[edge_idxs]
    assert (edge_idxs_reduced >= 0).all()
    return edge_complex_reduced, edge_idxs_reduced

@typechecked
def reduce_edge_complex_identity(base_dataset: BaseDataset, edge_idxs: TensorType['num_edges']) -> Tuple[EdgeComplex, TensorType['num_edges']]:
    """ Does not reduce the edge complex, i.e. an identity mapping. """
    return base_dataset.edge_complex(), edge_idxs


class TrajectoryForecastingDataset(Dataset):
    """ A dataset that does trajectory forecasting by masking parts of the trajectory."""
    
    def __init__(self, base_dataset: BaseDataset, config: DataConfig, rng: np.random.Generator,
                 idxs_subset: Iterable[int] | None = None):
        super().__init__()
        self.base_dataset = base_dataset
        logging.info('Building dataset edge complex')
        self.edge_complex = base_dataset.edge_complex()
        self.num_unmasked_edges = config.num_unmasked_edges
        self.min_number_masked_edges = config.min_number_masked_edges
        self.max_number_masked_edges = config.max_number_masked_edges
        self.edge_to_trajectory_adjacency_features_from_data = config.edge_to_trajectory_adjacency_features_from_data
        self.compute_cell_complex = config.compute_cell_complex
        self.reduce_instances_to = config.reduce_instances_to
        self.cached = config.cached
        self.edge_complex_reduction_fn = reduce_edge_complex_identity
        self.displacement_error_distance_metrics = config.displacement_error_distance_metrics
        self.sort_by = config.sort_by
        
        # Only select instances with enough edges such that i) enough edges can be masked, ii) the minimial edge length is respected
        min_num_edges = max(2,
                            config.min_number_edges if config.min_number_edges else 1,
                            config.num_unmasked_edges + 1 if config.num_unmasked_edges is not None else 1)
        
        assert min_num_edges > 1, 'The number of edges per instance should be larger than 1 to allow masking'
        if idxs_subset is None:
            idxs_subset = range(len(base_dataset))
        self._idxs = np.array([idx for idx in idxs_subset if base_dataset[idx]['edge_idxs'].size(0) >= min_num_edges])
        if config.size is not None:
            self._idxs = self._idxs[ : config.size] # Should only be used for debugging
        self._rng = rng
        
        # TODO: Does it make sense to also re-randomize test and val instances? 
        # If not, the split logic needs to be merged into a different module smh
        self.randomize_masked_edge_idxs()
        
        if self.compute_cell_complex:
            base_dataset.compute_cell_complex()

        self.cache = dict()
            
    def _get_reduced_edge_complex(self, idx: int) -> Tuple[EdgeComplex, TensorType['trajectory_length']]:
        """ Gets the edge complex after applying the reduction function `self.edge_complex_reduction_fn`. """
        if idx in self.cache:
            return self.cache[idx]
        else:
            trajectory = self.base_dataset[idx]
            edge_complex, edge_idxs = self.edge_complex_reduction_fn(self.base_dataset, trajectory['edge_idxs'])
            if self.cached:
                self.cache[idx] = edge_complex, edge_idxs
            return edge_complex, edge_idxs
          
    def randomize_masked_edge_idxs(self):
        
        sizes = np.array([self.base_dataset[idx]['edge_idxs'].size(0) for idx in self._idxs])
        p = self._rng.uniform(0., 1., size=len(self._idxs))
        
        if self.num_unmasked_edges is not None:
            lower = sizes - self.num_unmasked_edges
        elif self.min_number_masked_edges is None:
            lower = np.ones_like(sizes)
        elif isinstance(self.min_number_masked_edges, int):
            lower = np.ones_like(sizes) * self.min_number_masked_edges
        elif isinstance(self.min_number_masked_edges, float):
            lower = np.floor(sizes * self.min_number_masked_edges)
        else:
            raise ValueError(f'Cant comprehend min_number_masked_edges {self.min_number_masked_edges}')
        
        assert (lower < sizes).all(), f'The minimum number of edges to mask should be strictly smaller than the size of each instance'
        
        if self.num_unmasked_edges is not None:
            upper = sizes - self.num_unmasked_edges
        elif self.max_number_masked_edges is None:
            upper = sizes - 1
        elif isinstance(self.max_number_masked_edges, int):
            upper = np.ones_like(sizes) * self.max_number_masked_edges
        elif isinstance(self.max_number_masked_edges, float):
            upper = np.floor(sizes * self.max_number_masked_edges)
        else:
            raise ValueError(f'Cant comprehend max_number_masked_edges {self.max_number_masked_edges}')
        
        lower = np.maximum(lower, 1)
        upper = np.minimum(upper, sizes - 1)
        self._num_edges_to_mask = torch.from_numpy(np.round(lower + (upper - lower) * p).astype(int)).long()
        self.sort_idxs()
    
    def sort_idxs(self):
        if self.sort_by == 'num_masked':
            order = np.argsort(self._num_edges_to_mask.numpy())
        elif self.sort_by is None:
            return
        else:
            raise RuntimeError(f'Unsupported index sorting {self.sort_by}')
        self._idxs = self._idxs[order]
        self._num_edges_to_mask = self._num_edges_to_mask[order]
    
    def __getitem__(self, idx: int) -> TrajectoryForecastingBatch:
        sample_idx = self._idxs[idx]
        edge_complex, edge_idxs = self._get_reduced_edge_complex(sample_idx)
        trajectory = self.base_dataset[sample_idx]
        idx_last = trajectory['edge_idxs'].size(0) - self._num_edges_to_mask[idx]
        
        if 'edge_features' in trajectory and self.edge_to_trajectory_adjacency_features_from_data:
            trajectory_edge_features = trajectory['edge_features'][ : idx_last]
        else:
            trajectory_edge_features = None
        
        trajectory_complex = TrajectoryComplex.from_edge_complex(edge_complex=edge_complex, 
                                            trajectory_edge_idxs=[edge_idxs[ : idx_last]],
                                            trajectory_edge_orientations=[trajectory['edge_orientation'][ : idx_last]],
                                            trajectory_edge_features=[trajectory_edge_features] if trajectory_edge_features is not None else None,
                                            trajectory_features=trajectory['features'].view(1, -1), directed=self.base_dataset.directed)
        
        # Ground truth: nodes and edges that were masked
        masked_edge_idxs, masked_edge_orientations = trajectory_complex.edge_complex.oriented_edge_idxs_to_edge_idxs(
            edge_idxs[idx_last : ],
            trajectory['edge_orientation'][idx_last : ], self.base_dataset.directed)
        
        masked_node_idxs = trajectory_complex.edge_complex.node_adjacency[1 - masked_edge_orientations, masked_edge_idxs]
        
        if self.compute_cell_complex: # extend the trajectory complex with cell-complex information
            trajectory_complex = CellComplex.from_trajectory_complex(trajectory_complex,
                                                                     edge_laplacian_lower_idxs=self.base_dataset.lower_edge_hodge_laplacian_idxs,
                                                                     edge_laplacian_lower_weights=self.base_dataset.lower_edge_hodge_laplacian_weights.view(-1),
                                                                     edge_laplacian_upper_idxs=self.base_dataset.upper_edge_hodge_laplacian_idxs,
                                                                     edge_laplacian_upper_weights=self.base_dataset.upper_edge_hodge_laplacian_weights.view(-1),)
            
        batch = TrajectoryForecastingBatch(trajectory_complex=trajectory_complex, masked_node_idxs=masked_node_idxs, masked_edge_idxs=masked_edge_idxs, 
                                          masked_edge_orientations=masked_edge_orientations, sample_idxs=[sample_idx])
        return batch
       
    def __len__(self) -> int: return len(self._idxs)
    
    def setup_reduce_edge_complex_fn(self, task: 'TrajectoryForecastingTask', which: str):
        """ Sets up how to reduce the instances (complexes) based on the model. """
        if self.reduce_instances_to is None:
            self.edge_complex_reduction_fn = reduce_edge_complex_identity
        elif self.reduce_instances_to == 'receptive_field':
            if not task.should_reduce_edge_complex_to(which, 'receptive_field'):
                self.edge_complex_reduction_fn = reduce_edge_complex_identity
            else:
                k = task.model.receptive_field
                if k is None:
                    warnings.warn(f'Reducing instance complexes to the model receptive field, which is infinite.')
                    self.edge_complex_reduction_fn = reduce_edge_complex_identity
                else:
                    self.edge_complex_reduction_fn = partial(reduce_edge_complex_to_k_hop_neighbourhood, k + 1)
        else:
            raise ValueError(f'Unknown reduction of instance edge complexes {self.reduce_instances_to}')
                
    def register_task(self, task: 'TrajectoryForecastingTask', which: str):
        """ Registers a model, e.g. sets the reduction function of individual instances. """
        self.setup_reduce_edge_complex_fn(task, which)
         
    @property
    def num_nodes(self) -> int: return self.base_dataset.num_nodes
    
    @property
    def num_edges(self) -> int: return self.base_dataset.num_edges
                
    @property
    def num_node_features(self) -> int | None: return self.base_dataset.num_node_features
    
    @property
    def num_edge_features(self) -> int | None: return self.base_dataset.num_edge_features
    
    @property
    def num_trajectory_features(self) -> int | None: return self.base_dataset.num_trajectory_features
    
    @property
    def num_node_to_edge_adjacency_features(self) -> int | None: return self.edge_complex.graphs[0].node_to_edge_adjacency_features.size(-1)
    
    @property
    def num_edge_to_trajectory_adjacency_features(self) -> int | None:
        if len(self.base_dataset) == 0:
            return None
        trajectory_edge_features = self.base_dataset[0].get('edge_features', None)
        if trajectory_edge_features is None or not self.edge_to_trajectory_adjacency_features_from_data:
            return 0
        else:
            return trajectory_edge_features.size(-1)

    @property
    def positive_weight_nodes(self) -> float:
        """Weight for positive samples to account for the imbalance of positive nodes  to negative ones.
        It is computed as #neg / #pos
        = (N - #pos) / #pos = N / #pos - 1
        = (sum_i num_nodes) / (sum_i #pos_i) - 1
        = { 1 / num_nodes * (sum_i num_nodes) } / {1 / num_nodes * sum_i #pos_i} - 1
        = (sum_i 1 ) / (sum_i #pos_i / num_nodes) - 1
        = num_samples /  (sum_i #pos_i / num_nodes) - 1
        
        Note: This neglects duplicate nodes that may be masked in one trajectory.
        
        Returns
        -------
        pos_weight_nodes : float
            The weight for positive nodes (i.e. masked ones)
        """
        return (len(self) / (self._num_edges_to_mask / self.edge_complex.num_nodes).sum().item()) - 1 

    @property
    def positive_weight_edges(self) -> float:
        """Weight for positive samples to account for imbalance of positive edges to negatives.

        Note: This neglects duplicate edges that may be masked in one trajectory.

        Returns
        -------
        pos_weight_edges : float
            The weight for positive edges.
        """
        return (len(self) / (self._num_edges_to_mask / self.edge_complex.num_edges).sum().item()) - 1
    
    @staticmethod
    def collate(*args, **kwargs): return TrajectoryForecastingBatch.collate(*args, **kwargs)
        
   