import torch
import torch_scatter

from torchtyping import TensorType
from typeguard import typechecked
from typing import Tuple, List
import attrs
import torch.nn.functional as F

from trajectory_gnn.data.base import TrajectoryComplex
from trajectory_gnn.data.base import *
from trajectory_gnn.data.simplicial_complex import collate_on_complex, split_on_complex
from trajectory_gnn.data.trajectory_forecasting import TrajectoryForecastingPrediction

@attrs.define(frozen=True)
class TrajectoryForecastingBatch:
    trajectory_complex: TrajectoryComplex
    masked_node_idxs: TensorType['num_masked']
    masked_edge_idxs: TensorType['num_masked']
    masked_edge_orientations: TensorType['num_masked']
    masked_batch_idx: TensorType['num_masked'] | None = None
    
    # original indices in the (unshuffled) base-dataset
    sample_idxs: List = attrs.field(factory=lambda: [-1])
    
    @property
    def batch_size(self) -> int:
        return self.trajectory_complex.batch_size
    
    @property
    def number_masked(self) -> TensorType['batch_size']:
        """ The number of instances masked for each instance in the batch. """
        return torch_scatter.scatter_add(torch.ones_like(self.masked_batch_idx), self.masked_batch_idx, dim_size=self.batch_size)
        
    
    @typechecked
    def resize_predicted_trajectories(self, predictions: List[TrajectoryForecastingPrediction]) -> Tuple[
        TensorType['num_predicted'], TensorType['num_predicted'], TensorType['num_predicted']]:
        """ Reshapes the predicted trajectories to match with the attributes `masked_node_idxs` and `masked_edge_idxs`

        Parameters
        ----------
        predictions : List[TrajectoryForecastingPrediction]
            The predictions for each step.

        Returns
        -------
        TensorType['num_predicted']
            All predicted node idxs.
        TensorType['num_predicted']
            All predicted edge idxs.
        TensorType['num_predicted']
            All predicted edge orientations.
        """
        assert self.trajectory_complex.num_trajectories == self.batch_size, 'Not supported to have multiple trajectories per batch'
        pred_nodes, pred_edges, pred_edge_orientations = torch.empty_like(self.masked_node_idxs), torch.empty_like(self.masked_edge_idxs), torch.empty_like(self.masked_edge_idxs)
        sampled_node_idxs = torch.stack([p.successor_candidate_node_idxs[p.sampled_successor_candidate_idxs] for p in predictions]).T
        sampled_edge_idxs = torch.stack([p.successor_candidate_edge_idxs[p.sampled_successor_candidate_idxs] for p in predictions]).T
        sampled_edge_orientations = torch.stack([p.successor_candidate_edge_orientations[p.sampled_successor_candidate_idxs] for p in predictions]).T
        for batch_idx, batch_size in enumerate(self.number_masked):
            pred_nodes[self.masked_batch_idx == batch_idx] = sampled_node_idxs[batch_idx, :batch_size]
            pred_edges[self.masked_batch_idx == batch_idx] = sampled_edge_idxs[batch_idx, :batch_size]
            pred_edge_orientations[self.masked_batch_idx == batch_idx] = sampled_edge_orientations[batch_idx, :batch_size]
        return pred_nodes, pred_edges, pred_edge_orientations
    
    @property
    def masked_and_unmasked_node_idxs(self) -> Tuple[TensorType['num_nodes_in_all_trajectories'], 
                                                     TensorType['num_nodes_in_all_trajectories']]:
        """ Gets all masked and unmasked nodes in the trajectories. """ 
        edge_idxs, edge_orientations, trajectory_idxs = self.masked_and_unmasked_edge_idxs
        node_adjacency_idxs = self.trajectory_complex.edge_complex.edge_idxs_to_node_adjacency_idxs(edge_idxs)
        node_endpoint_idxs = self.trajectory_complex.edge_complex.node_adjacency[1 - edge_orientations, node_adjacency_idxs]
        # also find the startpoints: the edge sequence is expected to be ordered, i.e. per trajectory, the first one is
        # the starting edge
        _, argmin = torch_scatter.scatter_min(torch.arange(trajectory_idxs.size(0), device=trajectory_idxs.device, dtype=int),
                                                                 trajectory_idxs, 
                                                                 out=torch.full((self.trajectory_complex.num_trajectories,), trajectory_idxs.size(0), 
                                                                                dtype=int, device=trajectory_idxs.device))
        argmin = argmin[argmin < trajectory_idxs.size(0)]
        node_adjacency_start_idxs = self.trajectory_complex.edge_complex.node_adjacency[edge_orientations[argmin], node_adjacency_idxs[argmin]]
        return torch.cat((node_adjacency_start_idxs, node_endpoint_idxs)), torch.cat((trajectory_idxs[argmin], trajectory_idxs))
    
    @property
    def masked_and_unmasked_edge_idxs(self) -> Tuple[TensorType[int, 'num_edges_in_all_trajectories'], 
                                                     TensorType[int, 'num_edges_in_all_trajectories'], 
                                                     TensorType[int, 'num_edges_in_all_trajectories']]:
        """ Returns the masked and unmasked edge indices.

        Returns
        -------
        edge_idxs: TensorType['trajectory_size']
            All edge indices of masked and unmasked trajectories.
        edge_orientation: TensorType['trajectory_size']
            All edge orientations of masked and unmasked trajectories.
        batch_idxs: TensorType['trajectory_size']
            Which trajectory each edge belongs to
        """
        edge_idxs = torch.cat((self.trajectory_complex.edge_to_trajectory_adjacency[0],
                               self.masked_edge_idxs))
        edge_orientation = torch.cat((self.trajectory_complex.edge_to_trajectory_orientation,
                                      self.masked_edge_orientations))
        trajectory_idxs = torch.cat((self.trajectory_complex.edge_to_trajectory_adjacency[1],
                                self.masked_trajectory_idxs[self.masked_batch_idx]))
        return edge_idxs, edge_orientation, trajectory_idxs
        
    
    
    @property
    def masked_trajectory_idxs(self) -> TensorType[int, 'batch_size']:
        """ Which trajectory idxs are partially masked, i.e. the ones to predict. 
        This assumes that within each batch, the first trajectory is the masked one. 
        Returns `num_trajectories` for all batches that do not have any trajectory.
        """
        num_trajectories_per_batch = self.trajectory_complex.num_trajectories_per_batch
        offsets = torch.cumsum(num_trajectories_per_batch, 0) - num_trajectories_per_batch
        offsets[num_trajectories_per_batch == 0] = self.trajectory_complex.num_trajectories
        return offsets
             
    @property
    def max_number_masked(self) -> int:
        """ How many steps are masked at most. """
        return self.number_masked.max().item()
    
    def __attrs_post_init__(self):
        if self.masked_batch_idx is None:
            object.__setattr__(self, 'masked_batch_idx', torch.zeros_like(self.masked_node_idxs))
    
    def step_idxs_to_unmask(self) -> TensorType['batch_size']:
        """ Gets the indices in `masked_*_idxs` that correspond to the next step in the trajectory (i.e. that corresond
        to the first masked element in each complex.)
        If a sample in the batch has no masked elements left, the corresponding index will be set to `masked_*_idxs.size(0)`
        """
        return torch_scatter.scatter_min(torch.arange(self.masked_batch_idx.size(0), dtype=int, device=self.masked_batch_idx.device), 
                                         self.masked_batch_idx, dim_size=self.batch_size,
                                         out=torch.ones(self.batch_size, dtype=int, device=self.masked_batch_idx.device) * self.masked_batch_idx.size(0))[0]
        
    def advance_step(self) -> 'TrajectoryForecastingBatch':
        """Advances the masked trajectories in this batch by one step, i.e. unmasks the next hop in each trajectory and adds it to the trajectories.

        Returns
        -------
        TrajectoryForecastingBatch
            _description_
        """
        idxs_to_unmask = self.step_idxs_to_unmask()
        idxs_to_unmask_mask = idxs_to_unmask < self.masked_batch_idx.size(0) # Only select elements in the batch that can be advanced (i.e. that have at least one masked xy)
        edge_idxs_to_unmask = self.masked_edge_idxs[idxs_to_unmask[idxs_to_unmask_mask]]
        edge_orientations_to_unmask = self.masked_edge_orientations[idxs_to_unmask[idxs_to_unmask_mask]]
        # append these edges to the respective trajectories
        edge_to_trajectory_adjacency = torch.stack((
            edge_idxs_to_unmask, torch.arange(idxs_to_unmask.size(0))[idxs_to_unmask_mask]
        ), dim=0)
        mask = torch.ones_like(self.masked_node_idxs, dtype=bool, device=self.masked_node_idxs.device)
        mask[idxs_to_unmask[idxs_to_unmask_mask]] = False
        return TrajectoryForecastingBatch(
            trajectory_complex=attrs.evolve(
                self.trajectory_complex,
                edge_to_trajectory_orientation = torch.cat((
                    self.trajectory_complex.edge_to_trajectory_orientation, edge_orientations_to_unmask,
                ), dim=0),
                edge_to_trajectory_adjacency = torch.cat((
                    self.trajectory_complex.edge_to_trajectory_adjacency, edge_to_trajectory_adjacency,
                ), dim=1)
                ),
            masked_node_idxs=self.masked_node_idxs[mask],
            masked_edge_idxs=self.masked_edge_idxs[mask],
            masked_batch_idx=self.masked_batch_idx[mask],
            masked_edge_orientations=self.masked_edge_orientations[mask],
        )
        
    @property
    def batch_size(self) -> int:
        return self.trajectory_complex.batch_size
    
    def _map_to_all_tensors(self, _func):
        object.__setattr__(self, 'trajectory_complex', _func(self.trajectory_complex))
        object.__setattr__(self, 'masked_node_idxs', _func(self.masked_node_idxs))
        object.__setattr__(self, 'masked_edge_idxs', _func(self.masked_edge_idxs))
        object.__setattr__(self, 'masked_edge_orientations', _func(self.masked_edge_orientations))
        if self.masked_batch_idx is not None:
            object.__setattr__(self, 'masked_batch_idx', _func(self.masked_batch_idx))
        
    def pin_memory(self) -> 'TrajectoryForecastingBatch':
        self._map_to_all_tensors(lambda tensor: tensor.pin_memory())
        return self
    
    def to(self, device):
        self._map_to_all_tensors(lambda tensor: tensor.to(device))
        return self
    
    def get_trajectory_targets_masked_idxs(self) -> TensorType['batch_size']:
        """ Returns the indices in `self.masked_*_idxs` of the actual endpoints. 
            If there is no masked index for one batch, `self.masked_*_idx.size(0)` is returned.
        """
        target_idxs = torch_scatter.scatter_max(torch.arange(self.masked_batch_idx.size(0), dtype=int, device=self.masked_batch_idx.device), 
                                         self.masked_batch_idx, dim_size=self.batch_size,
                                         out=torch.full((self.batch_size,), -1, dtype=int, device=self.masked_batch_idx.device))[0]
        target_idxs[target_idxs < 0] = self.masked_batch_idx.size(0)
        return target_idxs
        
    def get_trajectory_masked_lengths(self) -> TensorType['batch_size']:
        """ Returns how many elements (nodes / edges) are masked per batch. """
        return torch_scatter.scatter_add(torch.ones_like(self.masked_batch_idx, dtype=int), self.masked_batch_idx, dim_size=self.batch_size)
    
    def get_masked_trajectory_endpoints(self) -> Tuple[TensorType['num_nonempty_batches'], TensorType['num_nonempty_batches'], TensorType['num_nonempty_batches'],
                                                       TensorType['num_nonempty_batches'], TensorType['num_nonempty_batches']]:
        """ Gets the target node indices, edge indices, edge orientation and batch indices of all trajectories that are masked and not empty (i.e. at least one edge is masked). 
        
        Returns:
        --------
        node_idxs : TensorType['num_nonempty_batches']
            Node indices of the endpoints in the masked trajectories.
        edge_idxs : TensorType['num_nonempty_batches']
            Edge indices of the endpoints in the masked trajectories.
        edge_orientations : TensorType['num_nonempty_batches']
            Edge orientations of the endpoints in the masked trajectories.
        step_idx : TensorType['num_nonempty_batches']
            The index in the per-trajectory count. E.g. if the k-th trajectory has t masked steps, then t - 1 will be returned,
            as it's the index of this element within the individual masked trajectory.
        batch_idx : TensorType['num_nonempty_batches']
            Which batch each index belongs to.
        
        """
        target_idxs = self.get_trajectory_targets_masked_idxs()
        target_idxs = target_idxs[target_idxs < self.masked_batch_idx.size(0)]
        
        lengths = self.get_trajectory_masked_lengths()
        lengths = lengths[lengths > 0] - 1
        
        return self.masked_node_idxs[target_idxs], self.masked_edge_idxs[target_idxs], self.masked_edge_orientations[target_idxs], \
            lengths, self.masked_batch_idx[target_idxs]
        
    
    @staticmethod
    def collate(batch: List['TrajectoryForecastingBatch']) -> 'TrajectoryForecastingBatch':
        """Collates multiple trajectory complexes into a batch.

        Returns
        -------
        batch : TrajectoryComplex
            The aggregated trajectory complex.
        masked_node_idxs : Tuple[tensor, shape [num_maksed_nodes]]
            The node indices that were masked in this batch and which batch they correspond to.
        masked_edge_idxs : Tuple[tensor, shape [num_masked_edges]]
            The edge indices that were masked in this batch and which batch they correspond to.
        """
        if len(batch) == 1:
            return batch[0]
        
        collated_complex = batch[0].trajectory_complex.collate([b.trajectory_complex for b in batch])
        collated_masked_node_idxs, masked_batch_idx = collate_on_complex([b.masked_node_idxs for b in batch], 
                                                                                        collated_complex.edge_complex.node_batch_idxs,
                                                                                        feature_batch_idxs=[b.masked_batch_idx for b in batch],
                                                                                        batch_sizes=torch.tensor([b.batch_size for b in batch]))
        collated_masked_edge_idxs, _ = collate_on_complex([b.masked_edge_idxs for b in batch], 
                                                                                        collated_complex.edge_complex.edge_batch_idxs,
                                                                                        feature_batch_idxs=[b.masked_batch_idx for b in batch],
                                                                                        batch_sizes=torch.tensor([b.batch_size for b in batch]))
        
        return TrajectoryForecastingBatch(trajectory_complex=collated_complex, masked_node_idxs=collated_masked_node_idxs,
                                          masked_edge_idxs=collated_masked_edge_idxs, masked_batch_idx=masked_batch_idx,
                                          masked_edge_orientations=torch.cat([b.masked_edge_orientations for b in batch]),
                                          sample_idxs=sum((b.sample_idxs for b in batch), start=[]))
    
    @staticmethod
    def split(collated: 'TrajectoryForecastingBatch') -> \
                List['TrajectoryForecastingBatch']:
        """Inverse operation to `collate`.

        Parameters:
        -----------
        collated : TrajectoryForecastingBatch
            The collated batch to split

        Returns
        -------
        result : List[TrajectoryForecastingBatch]
            The separated batches in this batch.
        """
        if collated.batch_size == 1:
            return [collated]
        
        return [TrajectoryForecastingBatch(trajectory_complex=complex, masked_node_idxs=masked_node_idxs,
                                           masked_edge_idxs=masked_edge_idxs, masked_edge_orientations=masked_edge_orientation,
                                           sample_idxs=[sample_idx]) 
                for (complex, masked_node_idxs, masked_edge_idxs, masked_edge_orientation, sample_idx) in zip(
            collated.trajectory_complex.split(collated.trajectory_complex),
            split_on_complex(
                collated.masked_node_idxs, 
                collated.masked_batch_idx, 
                element_batch_idxs=collated.trajectory_complex.edge_complex.node_batch_idxs,
                batch_size=collated.batch_size),
            split_on_complex(
                collated.masked_edge_idxs, 
                collated.masked_batch_idx, 
                element_batch_idxs=collated.trajectory_complex.edge_complex.edge_batch_idxs,
                batch_size=collated.batch_size),
            [collated.masked_edge_orientations[collated.masked_batch_idx == batch_idx] for batch_idx in range(collated.batch_size)],
            collated.sample_idxs,
            )]
