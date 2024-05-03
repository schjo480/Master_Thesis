import torch
import torch_scatter

from torchtyping import TensorType
from typing import List
import attrs
from typeguard import typechecked

from trajectory_gnn.utils.utils import scatter_normalize_positive
from trajectory_gnn.utils import map_to_optional_tensors
from trajectory_gnn.data.base import TrajectoryComplex

@attrs.define()
class TrajectoryForecastingPrediction:
    """ Wraps trajectory forecasting results""" 

    embeddings : List[List[TensorType | None]] | None = None
    # Independent probabilities of observing an element. If not given, derived from logits
    probabilities : List[TensorType | None] | None = None
    
    # Successor sampling: The ordering is important, i.e. `successor_candidate_node_idxs`
    # `successor_candidate_edge_idxs`, `successor_candidate_edge_orientations`,
    # `successor_candidate_trajectory_idxs` are supposed to be zippable
    successor_candidate_node_idxs: TensorType['num_successor_candidates'] | None = None
    successor_candidate_edge_idxs: TensorType['num_successor_candidates'] | None = None
    successor_candidate_edge_orientations: TensorType['num_successor_candidates'] | None = None
    successor_candidate_trajectory_idxs: TensorType ['num_successor_candidates']| None = None # Kinda like batch idxs: For each candidate successor, which trajectory it belongs to
    sampled_successor_candidate_idxs: TensorType['num_trajectories'] | None = None # Which candidates were actually sampled
    successor_candidate_node_map: TensorType['num_nodes'] | None = None
    successor_candidate_edge_map: TensorType['num_edges'] | None = None
    
    # Probabilities for each potential successor per order. If not given, derived from logits
    successor_candidate_logits: List[TensorType['num_successor_candidates'] | None] | None = None
    successor_candidate_node_probabilities: List[TensorType['num_successor_candidates'] | None] | None = None
    successor_candidate_edge_probabilities: List[TensorType['num_successor_candidates'] | None] | None = None
    
    # Additional attributes that can be predicted by some models only
    node_random_walk_weights: TensorType['num_edges'] | None = None # only logged by Gretel

    def _map_to_optional_tensors(self, _func):
        """ Maps a function to all tensors of this dataclass """
        self.embeddings=map_to_optional_tensors(_func, self.embeddings)
        self.probabilities=map_to_optional_tensors(_func, self.probabilities)
        self.successor_candidate_node_idxs=map_to_optional_tensors(_func, self.successor_candidate_node_idxs)
        self.successor_candidate_edge_idxs=map_to_optional_tensors(_func, self.successor_candidate_edge_idxs)
        self.successor_candidate_edge_orientations=map_to_optional_tensors(_func, self.successor_candidate_edge_orientations)
        self.successor_candidate_trajectory_idxs=map_to_optional_tensors(_func, self.successor_candidate_trajectory_idxs)
        self.sampled_successor_candidate_idxs=map_to_optional_tensors(_func, self.sampled_successor_candidate_idxs)
        self.successor_candidate_node_map=map_to_optional_tensors(_func, self.successor_candidate_node_map)
        self.successor_candidate_edge_map=map_to_optional_tensors(_func, self.successor_candidate_edge_map)
        self.successor_candidate_node_probabilities=map_to_optional_tensors(_func, self.successor_candidate_node_probabilities)
        self.successor_candidate_edge_probabilities=map_to_optional_tensors(_func, self.successor_candidate_edge_probabilities)
        self.successor_candidate_logits=map_to_optional_tensors(_func, self.successor_candidate_logits)

    @typechecked
    def to(self, device) -> 'TrajectoryForecastingPrediction':
        self._map_to_optional_tensors(lambda tensor: tensor.to(device))
        return self

    @typechecked
    def detach(self) -> 'TrajectoryForecastingPrediction':
        self._map_to_optional_tensors(lambda tensor: tensor.detach())
        return self
    
    def populate_successor_candidates_from_batch(self, batch: TrajectoryComplex, recompute: bool=False):
        """ Populates the successor candidates from a trajectory complex batch. """
        if self.successor_candidate_node_idxs is None or recompute: # `batch.get_advance_successors` is rather expensive, do it only once
            self.successor_candidate_node_idxs, self.successor_candidate_edge_idxs, self.successor_candidate_edge_orientations, \
                    self.successor_candidate_trajectory_idxs, self.successor_candidate_node_map, \
                    self.successor_candidate_edge_map = batch.get_advance_successors()
    
    @property
    def num_embeddings(self) -> int:
        if self.embeddings is None:
            return 0
        else:
            return len(self.embeddings)
    
    @typechecked
    def get_embedding(self, layer_num: int, order: int) -> TensorType['num_order', -1] | None:
        """ Gets the embeddings of elements of one order at a given layer. Use -1 for the logit layer. """
        if self.embeddings is not None and order < len(self.embeddings[layer_num]):
            return self.embeddings[layer_num][order]
        else:
            return None    
    
    def get_embeddings(self, order: int) -> List[TensorType['num_order', -1] | None] | None:
        """ Gets embeddings for elements of an order for all layers. The last one corresponds to the logits """
        if self.embeddings is not None:
            return [self.get_embedding(layer_num, order) for layer_num in range(len(self.embeddings))]
        else:
            return None
    
    @typechecked
    def get_logits(self, order: int) -> TensorType['num_order'] | None:
        """ Gets the logits for elements of an order """
        emb = self.get_embedding(-1, order)
        if emb is not None:
            assert emb.size(-1) == 1, f'logits should be one dimensional, not of size {emb.size()}'
            emb = emb.view(-1) # embeddings are of size [num_order, d]
        return emb
    
    @typechecked
    def get_probabilities(self, order: int) -> TensorType['num_order'] | None:
        """ Gets probabilities of each element of an order independently (i.e. sigmoid instead of softmax normalization). """
        if self.probabilities is not None and order < len(self.probabilities) and self.probabilities[order] is not None:
            return self.probabilities[order]
        logits = self.get_logits(order)
        if logits is not None:
            return torch.sigmoid(logits)
        else:
            return None
        
    @typechecked
    def get_successor_candidate_logits(self, order: int) -> TensorType['num_successor_candidates'] | None:
        """ Gets the logits of all successor candidates of an order. """
        if self.successor_candidate_logits is not None and self.successor_candidate_logits[order] is not None:
            return self.successor_candidate_logits[order]
        logits = self.get_logits(order)
        if logits is None:
            return None
        if order == 0:
            return logits[self.successor_candidate_node_idxs]
        elif order == 1:
            return logits[self.successor_candidate_edge_idxs]
        else:
            return None
        
    @typechecked
    def get_successor_candidate_probabilities(self, order: int) -> TensorType['num_successor_candidates'] | None:
        """ Gets the probabilities assigned to successor candidates. The priority in which it is looked up is:
        1. `self.successor_candidate_*_probabilities` : dedicated probabilities that can be set by the model
        2. `softmax(self.get_successor_candidate_logits)) : probabilities computed by a softmax over the model's outputted logits
        3. `renormalize(self.probabilities(order))` : renormalization of per-instance probabilities on the whole complex
        
        """
        probs = {
            0 : self.successor_candidate_node_probabilities,
            1 : self.successor_candidate_edge_probabilities,
            }.get(order, None)
        if probs is not None:
            return probs
        
        # Derive from softmax of logits
        logits = self.get_successor_candidate_logits(order)
        if logits is not None:
            return torch_scatter.scatter_softmax(logits, self.successor_candidate_trajectory_idxs)
        
        # Derive from instance-wise independent probabilities (TODO: is this a good idea?)
        probabilities = self.get_probabilities(order)
        if probabilities is not None:
            probabilities = probabilities[{
                0 : self.successor_candidate_node_idxs,
                1 : self.successor_candidate_edge_idxs,
            }.get(order, None)]
            return scatter_normalize_positive(probabilities, self.successor_candidate_trajectory_idxs)
        
        return None
        
    @typechecked
    def get_sampled_successor_candidate_idxs_by_max_score(self, by_order: int, 
                                                          dim_size: int | None = None) -> TensorType['num_trajectories'] | None:
        """ Gets indices to index successor candidates that were sampled based on a proxy.
        
        Paramters:
        ----------
        by_order : int
            Based on the scores / logits of which order to select the successor.
        dim_size : int | None, optional
            For how many to select sucessors.

        Returns
        -------
        idxs : TensorType['num_trajectories']
            For each trajectory, which successor candidate to pick.
        """
        if self.sampled_successor_candidate_idxs is not None:
            return self.sampled_successor_candidate_idxs
        # Get a proxy to determine the sampled candidate
        proxy = self.get_successor_candidate_logits(by_order)
        if proxy is None:
            proxy = self.get_successor_candidate_probabilities(by_order)
        if proxy is None:
            return None
        _, argmax = torch_scatter.scatter_max(proxy.view(-1), 
                                              self.successor_candidate_trajectory_idxs,
                                              dim_size=dim_size)
        return argmax
