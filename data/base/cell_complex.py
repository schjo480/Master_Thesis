import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict
from torchtyping import TensorType

from typeguard import typechecked

import torch
import torch_scatter
import attrs
from trajectory_gnn.data.base import TrajectoryComplex, EdgeComplex

@attrs.define()
class CellComplex(TrajectoryComplex):
    
    """ Also models cells, i.e. splits the edge adjacency into upper and lower hodge laplacians. """

    edge_laplacian_lower_idxs: TensorType[2, 'num_lower_adj'] = None
    edge_laplacian_lower_weights: TensorType['num_lower_adj'] = None
    edge_laplacian_upper_idxs: TensorType[2, 'num_upper_adj'] = None
    edge_laplacian_upper_weights: TensorType['num_upper_adj'] = None
    
    @staticmethod
    @typechecked
    def from_trajectory_complex(trajectory_complex: TrajectoryComplex,
                                  edge_laplacian_lower_idxs: TensorType[2, 'num_lower_adj'],
                                    edge_laplacian_lower_weights: TensorType['num_lower_adj'],
                                    edge_laplacian_upper_idxs: TensorType[2, 'num_upper_adj'],
                                    edge_laplacian_upper_weights: TensorType['num_upper_adj'],) -> 'CellComplex':
        """ Extends a trajectory complex with cell adjacency information. """
        return CellComplex(
            edge_complex=trajectory_complex.edge_complex,
            trajectory_features=trajectory_complex.trajectory_features,
            edge_to_trajectory_adjacency=trajectory_complex.edge_to_trajectory_adjacency,
            edge_to_trajectory_orientation=trajectory_complex.edge_to_trajectory_orientation,
            edge_to_trajectory_adjacency_features=trajectory_complex.edge_to_trajectory_adjacency_features,
            trajectory_adjacency=trajectory_complex.trajectory_adjacency,
            trajectory_adjacency_features=trajectory_complex.trajectory_adjacency_features,
            attributes=trajectory_complex.attributes,
            trajectory_batch_idxs=trajectory_complex.trajectory_batch_idxs,
            
            edge_laplacian_lower_idxs=edge_laplacian_lower_idxs,
            edge_laplacian_lower_weights=edge_laplacian_lower_weights,
            edge_laplacian_upper_idxs=edge_laplacian_upper_idxs,
            edge_laplacian_upper_weights=edge_laplacian_upper_weights,
        )

    def _map_to_all_tensors(self, _func):
        super()._map_to_all_tensors(_func)
        object.__setattr__(self, 'edge_laplacian_lower_idxs', _func(self.edge_laplacian_lower_idxs))
        object.__setattr__(self, 'edge_laplacian_lower_weights', _func(self.edge_laplacian_lower_weights))
        object.__setattr__(self, 'edge_laplacian_upper_idxs', _func(self.edge_laplacian_upper_idxs))
        object.__setattr__(self, 'edge_laplacian_upper_weights', _func(self.edge_laplacian_upper_weights))
    
    @staticmethod
    def collate_internal(batch: List['CellComplex']) -> Dict:
        kwargs_trajectory_complex = TrajectoryComplex.collate_internal(batch)
        
        edge_batch_idxs = kwargs_trajectory_complex['edge_complex'].edge_batch_idxs
         # compute how many edges are in each batch in `batches`
        num_edges_collated = torch_scatter.scatter_add(torch.ones_like(edge_batch_idxs), edge_batch_idxs, dim=0)
        num_instances_per_batch = torch.tensor([b.batch_size for b in batch])
        num_edges_collated = torch.tensor([num_edges_collated[end - num_batches : end].sum().item() for num_batches, end in 
                             zip(num_instances_per_batch, torch.cumsum(num_instances_per_batch, 0))])
        edge_offsets = torch.cumsum(num_edges_collated, 0) - num_edges_collated
        
        edge_laplacian_lower_idxs = torch.cat([b.edge_laplacian_lower_idxs + edge_offsets[batch_idx]
                                               for batch_idx, b in enumerate(batch)], dim=1)
        edge_laplacian_upper_idxs = torch.cat([b.edge_laplacian_upper_idxs + edge_offsets[batch_idx]
                                               for batch_idx, b in enumerate(batch)], dim=1)
        edge_laplacian_lower_weights = torch.cat([b.edge_laplacian_lower_weights for b in batch])
        edge_laplacian_upper_weights = torch.cat([b.edge_laplacian_upper_weights for b in batch])
        
        return kwargs_trajectory_complex | dict(
            edge_laplacian_lower_idxs=edge_laplacian_lower_idxs,
            edge_laplacian_upper_idxs=edge_laplacian_upper_idxs,
            edge_laplacian_lower_weights=edge_laplacian_lower_weights,
            edge_laplacian_upper_weights=edge_laplacian_upper_weights,)
    
    @staticmethod
    def collate(batch: List['CellComplex']) -> 'CellComplex':
        if len(batch) == 1:
            return batch[0]
        return CellComplex(**CellComplex.collate_internal(batch))
    
    @staticmethod
    def split_internal(batch: 'CellComplex') -> List[Dict]:
        trajectory_complex_kwargs = TrajectoryComplex.split_internal(batch)
        num_edges = torch_scatter.scatter_add(torch.ones_like(batch.edge_complex.edge_batch_idxs), batch.edge_complex.edge_batch_idxs,
                                              dim_size=len(trajectory_complex_kwargs))
        edge_offsets = torch.cumsum(num_edges, 0) - num_edges
        
        lower_idxs, upper_idxs, lower_weights, upper_weights = [], [], [], []
        for edge_offset_batch, num_edges_batch in zip(edge_offsets, num_edges):
            mask_lower = ((batch.edge_laplacian_lower_idxs >= edge_offset_batch ) & (batch.edge_laplacian_lower_idxs < edge_offset_batch + num_edges_batch)).all(0)
            lower_idxs.append(batch.edge_laplacian_lower_idxs[:, mask_lower] - edge_offset_batch)
            lower_weights.append(batch.edge_laplacian_lower_weights[mask_lower])
            
            mask_upper = ((batch.edge_laplacian_upper_idxs >= edge_offset_batch ) & (batch.edge_laplacian_upper_idxs < edge_offset_batch + num_edges_batch)).all(0)
            upper_idxs.append(batch.edge_laplacian_upper_idxs[:, mask_upper] - edge_offset_batch)
            upper_weights.append(batch.edge_laplacian_upper_weights[mask_upper])
        
        return [kwargs | dict(
            edge_laplacian_lower_idxs=lower_idxs,
            edge_laplacian_lower_weights=lower_weights,
            edge_laplacian_upper_idxs=upper_idxs,
            edge_laplacian_upper_weights=upper_weights,
            ) for (kwargs, lower_idxs, lower_weights, upper_idxs, upper_weights) in zip(trajectory_complex_kwargs, lower_idxs, lower_weights, upper_idxs, upper_weights)]
        
        
    @staticmethod
    def split(batch: 'CellComplex') -> List['CellComplex']:
        """ Inverse of `collate` """
        if batch.batch_size == 1:
            return [batch]
        return [CellComplex(**kwargs) for kwargs in CellComplex.split_internal(batch)]
       
    @staticmethod
    def compute_cell_adjacency(edge_complex: EdgeComplex,
                          cell_edge_idxs: List[List[int]],
                          cell_edge_orientations: List[List[int]]) -> Tuple[
        TensorType[2, 'num_edge_to_cell_adjacency'],
        TensorType['num_edge_to_cell_adjacency'],
        TensorType[2, 'num_cell_adjacency']
    ]:
        """ Computes attributes of a CellComplex based on the edges of an EdgeComplex
        and precomputed faces of a plane embedding of said EdgeComplex. Adjacency between
        cells is computed iff they share some edge in `cell_edge_idxs`

        Parameters:
        -----------
        edge_complex : EdgeComplex
            The edge complex to base faces of.
        cell_edge_idxs : List[List[int]]
            For each cell, which edge idxs it is comprised of
        cell_edge_orientations : List[List[int]]
            For each cell, how the edges it is comprised of are oriented

        Returns
        -------
        edge_to_cell_adjacency : TensorType[2, 'num_edge_to_cell_adj']
            Adjacency from edges to cells
        edge_to_cell_adjacency_orientation : TensorType['num_edge_to_cell_adj']
            Adjacency orientation from edges to cells
        cell_adjacency : TensorType[2, 'num_cell_adj']
            Adjacency between cells
        """
        assert edge_complex.batch_size == 1, f'cells should be computed on an uncollated complex'

        col = torch.cat([torch.tensor(edge_idxs) for edge_idxs in cell_edge_idxs])
        row = torch.cat([torch.ones(len(edge_idxs), dtype=int) * cell_idx for (cell_idx, edge_idxs) in enumerate(cell_edge_idxs)])
        
        edge_to_cell_adjacency = torch.stack((col, row))
        edge_to_cell_orientation = torch.cat([torch.tensor(orient) for orient in cell_edge_orientations])
        
        # constructs cell adjacency: cells are adjacenct iff they share an edge
        cell_indicator = sp.coo_matrix((np.ones(row.size(0)), (row, col)), shape=(len(cell_edge_idxs), edge_complex.num_edges)) # C, E
        cell_adjacency = (cell_indicator @ cell_indicator.T).tolil()
        cell_adjacency.setdiag(0)
        cell_adjacency = torch.from_numpy(np.array(cell_adjacency.nonzero())).long()
    
        return edge_to_cell_adjacency, edge_to_cell_orientation, cell_adjacency


# @attrs.define()
# class CellComplex(TrajectoryComplex):
#     """ Models cells as co-chains. """
#     edge_to_cell_adjacency: TensorType[2, 'num_edges_in_cells'] = attrs.field(
#         validator=lambda cc, attr, value: value.size(1) == cc.edge_to_cell_adjacency_orientation.size(0),
#         default=None,
#     )
#     edge_to_cell_adjacency_orientation: TensorType['num_edges_in_cells'] = None
#     cell_adjacency: TensorType[2, 'num_inter_cell_adj'] = None
    
#     cell_feature_names: List[str] | None = None
#     cell_features: TensorType['num_cells', 'num_cell_features'] | None = None
    
#     cell_batch_idxs: TensorType['num_cells'] = None
    
#     def __attrs_post_init__(self):
#         super().__attrs_post_init__()
#         if self.cell_batch_idxs is None:
#             object.__setattr__(self, 'cell_batch_idxs', 
#                                torch.zeros(self.num_cells, dtype=int, device=self.edge_to_cell_adjacency.device))
    
#     @property
#     def num_cells(self) -> int:
#         if self.cell_features is None:
#             candidates = [-1]
#             if self.cell_adjacency.numel() > 0:
#                 candidates.append(self.cell_adjacency.max().item())
#             if self.edge_to_cell_adjacency.numel() > 0:
#                 candidates.append(self.edge_to_cell_adjacency[1].max().item())
#             return max(candidates) + 1
#         else:
#             return self.cell_features.size(0)
        
#     @staticmethod
#     def compute_cell_adjacency(edge_complex: EdgeComplex,
#                           cell_edge_idxs: List[List[int]],
#                           cell_edge_orientations: List[List[int]]) -> Tuple[
#         TensorType[2, 'num_edge_to_cell_adjacency'],
#         TensorType['num_edge_to_cell_adjacency'],
#         TensorType[2, 'num_cell_adjacency']
#     ]:
#         """ Computes attributes of a CellComplex based on the edges of an EdgeComplex
#         and precomputed faces of a plane embedding of said EdgeComplex. Adjacency between
#         cells is computed iff they share some edge in `cell_edge_idxs`

#         Parameters:
#         -----------
#         edge_complex : EdgeComplex
#             The edge complex to base faces of.
#         cell_edge_idxs : List[List[int]]
#             For each cell, which edge idxs it is comprised of
#         cell_edge_orientations : List[List[int]]
#             For each cell, how the edges it is comprised of are oriented

#         Returns
#         -------
#         edge_to_cell_adjacency : TensorType[2, 'num_edge_to_cell_adj']
#             Adjacency from edges to cells
#         edge_to_cell_adjacency_orientation : TensorType['num_edge_to_cell_adj']
#             Adjacency orientation from edges to cells
#         cell_adjacency : TensorType[2, 'num_cell_adj']
#             Adjacency between cells
#         """
#         assert edge_complex.batch_size == 1, f'cells should be computed on an uncollated complex'

#         col = torch.cat([torch.tensor(edge_idxs) for edge_idxs in cell_edge_idxs])
#         row = torch.cat([torch.ones(len(edge_idxs), dtype=int) * cell_idx for (cell_idx, edge_idxs) in enumerate(cell_edge_idxs)])
        
#         edge_to_cell_adjacency = torch.stack((col, row))
#         edge_to_cell_orientation = torch.cat([torch.tensor(orient) for orient in cell_edge_orientations])
        
#         # constructs cell adjacency: cells are adjacenct iff they share an edge
#         cell_indicator = sp.coo_matrix((np.ones(row.size(0)), (row, col)), shape=(len(cell_edge_idxs), edge_complex.num_edges)) # C, E
#         cell_adjacency = (cell_indicator @ cell_indicator.T).tolil()
#         cell_adjacency.setdiag(0)
#         cell_adjacency = torch.from_numpy(np.array(cell_adjacency.nonzero())).long()
    
#         return edge_to_cell_adjacency, edge_to_cell_orientation, cell_adjacency
    
#     @staticmethod
#     def from_trajectory_complex(trajectory_complex: TrajectoryComplex,
#                                   edge_to_cell_adjacency: TensorType[2, 'num_edge_to_cell_adj'],
#                                   edge_to_cell_orientation: TensorType['num_edge_to_cell_adj'],
#                                   cell_adjacency: TensorType[2, 'num_cell_adjacency'],
#                                   cell_features: TensorType['num_cells', 'cell_dim'] | None = None,
#                                   cell_feature_names: List[str] | None = None) -> 'CellComplex':
#         """ Extends a trajectory complex with cell adjacency information. """
#         return CellComplex(
#             edge_complex=trajectory_complex.edge_complex,
#             trajectory_features=trajectory_complex.trajectory_features,
#             edge_to_trajectory_adjacency=trajectory_complex.edge_to_trajectory_adjacency,
#             edge_to_trajectory_orientation=trajectory_complex.edge_to_trajectory_orientation,
#             edge_to_trajectory_adjacency_features=trajectory_complex.edge_to_trajectory_adjacency_features,
#             trajectory_adjacency=trajectory_complex.trajectory_adjacency,
#             trajectory_adjacency_features=trajectory_complex.trajectory_adjacency_features,
#             attributes=trajectory_complex.attributes,
#             trajectory_batch_idxs=trajectory_complex.trajectory_batch_idxs,
            
#             edge_to_cell_adjacency=edge_to_cell_adjacency,
#             edge_to_cell_adjacency_orientation=edge_to_cell_orientation,
#             cell_adjacency=cell_adjacency,
#             cell_features=cell_features,
#             cell_feature_names=cell_feature_names
#         )
        
#     def _map_to_all_tensors(self, _func):
#         super()._map_to_all_tensors(_func)
#         if self.cell_features is not None:
#             object.__setattr__(self, 'cell_features', _func(self.cell_features))
#         object.__setattr__(self, 'cell_adjacency', _func(self.cell_adjacency))
#         object.__setattr__(self, 'edge_to_cell_adjacency', _func(self.edge_to_cell_adjacency))
#         object.__setattr__(self, 'edge_to_cell_adjacency_orientation', _func(self.edge_to_cell_adjacency_orientation))
#         object.__setattr__(self, 'cell_batch_idxs', _func(self.cell_batch_idxs))
    
#     @staticmethod
#     def collate_internal(batch: List['CellComplex']) -> Dict:
#         kwargs_trajectory_complex = TrajectoryComplex.collate_internal(batch)
#         edge_batch_idxs = kwargs_trajectory_complex['edge_complex'].edge_batch_idxs
        
#         if all(b.cell_features is not None for b in batch):
#             cell_features = torch.cat([b.cell_features for b in batch])
#         else:
#             cell_features = None
        
#         num_cells = torch.tensor([b.num_cells for b in batch])
#         cell_offsets = torch.cumsum(num_cells, 0) - num_cells
#         num_instances_per_batch = torch.tensor([b.batch_size for b in batch])
        
#         batch_sizes = torch.tensor([b.batch_size for b in batch])
#         batch_offsets = torch.cumsum(batch_sizes, 0) - batch_sizes
        
#         # compute how many edges are in each batch in `batches`
#         num_edges_collated = torch_scatter.scatter_add(torch.ones_like(edge_batch_idxs), edge_batch_idxs, dim=0)
#         num_edges_collated = torch.tensor([num_edges_collated[end - num_batches : end].sum().item() for num_batches, end in 
#                              zip(num_instances_per_batch, torch.cumsum(num_instances_per_batch, 0))])
#         edge_offsets = torch.cumsum(num_edges_collated, 0) - num_edges_collated
        
#         edge_to_cell_adjacency = torch.cat([adj + torch.tensor([[edge_offset], [cell_offset]]) for adj, cell_offset, edge_offset in 
#                                             zip((b.edge_to_cell_adjacency for b in batch), cell_offsets, edge_offsets)], dim=-1)
#         edge_to_cell_adjacency_orientation = torch.cat([b.edge_to_cell_adjacency_orientation for b in batch])
#         cell_adjacency = torch.cat([b.cell_adjacency + offset for b, offset in zip(batch, cell_offsets)], dim=-1)
#         return dict(
#             cell_features=cell_features,
#             edge_to_cell_adjacency=edge_to_cell_adjacency,
#             edge_to_cell_adjacency_orientation=edge_to_cell_adjacency_orientation,
#             cell_batch_idxs=torch.cat([b.cell_batch_idxs + offset for b, offset in zip(batch, batch_offsets)]),
#             cell_adjacency=cell_adjacency) | kwargs_trajectory_complex
    
#     @staticmethod
#     def collate(batch: List['CellComplex']) -> 'CellComplex':
#         if len(batch) == 1:
#             return batch[0]
#         return CellComplex(**CellComplex.collate_internal(batch))
    
#     @staticmethod
#     def split_internal(batch: 'CellComplex') -> List[Dict]:
#         trajectory_complex_kwargs = TrajectoryComplex.split_internal(batch)
        
#         num_cells = torch_scatter.scatter_add(torch.ones_like(batch.cell_batch_idxs), batch.cell_batch_idxs,
#                                               dim_size=len(trajectory_complex_kwargs))
#         cell_offsets = torch.cumsum(num_cells, 0) - num_cells
#         num_edges = torch_scatter.scatter_add(torch.ones_like(batch.edge_complex.edge_batch_idxs), batch.edge_complex.edge_batch_idxs,
#                                               dim_size=len(trajectory_complex_kwargs))
#         edge_offsets = torch.cumsum(num_edges, 0) - num_edges
        
#         cell_features, cell_adjacencies, edge_to_cell_adjacencies, edge_to_cell_adjacency_orientations = [], [], [], []
#         for cell_offset, cell_size, edge_offset, edge_size in zip(cell_offsets, num_cells, edge_offsets, num_edges):
#             if batch.cell_features is None:
#                 cell_features.append(None)
#             else:
#                 cell_features.append(batch.cell_features[cell_offset : cell_offset + cell_size])
#             mask_cell_adjacencies = ((batch.cell_adjacency >= cell_offset) & (batch.cell_adjacency < (cell_offset + cell_size))).all(0) # num_cell_adj
#             cell_adjacencies.append(batch.cell_adjacency[:, mask_cell_adjacencies] - cell_offset)
            
#             mask_edge_to_cell_adjacencies = (batch.edge_to_cell_adjacency[1] >= cell_offset) & (batch.edge_to_cell_adjacency[1] < cell_offset + cell_size)
#             edge_to_cell_adjacency = batch.edge_to_cell_adjacency[:, mask_edge_to_cell_adjacencies]
#             edge_to_cell_adjacency[0] -= edge_offset
#             edge_to_cell_adjacency[1] -= cell_offset
#             edge_to_cell_adjacencies.append(edge_to_cell_adjacency)
#             edge_to_cell_adjacency_orientations.append(batch.edge_to_cell_adjacency_orientation[mask_edge_to_cell_adjacencies])
        
#         return [kwargs | dict(cell_adjacency=cell_adjacency, edge_to_cell_adjacency=edge_to_cell_adjacency,
#                             edge_to_cell_adjacency_orientation=edge_to_cell_adjacency_orientation,
#                             cell_features=cell_feature) for (kwargs, cell_adjacency, edge_to_cell_adjacency, edge_to_cell_adjacency_orientation,
#                                                              cell_feature) in zip(trajectory_complex_kwargs, cell_adjacencies,
#                                                                                   edge_to_cell_adjacencies, edge_to_cell_adjacency_orientations,
#                                                                                   cell_features)]
#     @staticmethod
#     def split(batch: 'CellComplex') -> List['CellComplex']:
#         """ Inverse of `collate` """
#         if batch.batch_size == 1:
#             return [batch]
#         return [CellComplex(**kwargs) for kwargs in CellComplex.split_internal(batch)]
       