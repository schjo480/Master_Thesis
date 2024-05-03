import attrs
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from trajectory_gnn.data.base import TrajectoryComplex
from torchtyping import TensorType, patch_typeguard
from typing import Iterable, List, Tuple
from pytorch_lightning.callbacks import Callback
from data.trajectory_forecasting import TrajectoryForecastingPrediction, TrajectoryForecastingBatch, TrajectoryForecastingDataset
from utils.utils import positional_encoding
from config import ModelConfig
from typeguard import typechecked

patch_typeguard()

class BaseModel(torch.nn.Module):
    """ Base model for trajectory tasks. """
    
    def __init__(self, config: ModelConfig, dataset: TrajectoryForecastingDataset):
        super().__init__()
    
    @property
    def needs_gradient_descent_training(self) -> bool:
        """If the model should be trained using a pytorch-lightning gradient descent framework.

        Returns
        -------
        bool
            If the model will trained using a pytorch-lightning trainer.
        """
        return True
    
    @property
    def name(self) -> str:
        return 'unnamed_model'

class TrajectoryForecasting(BaseModel):
    """ Base model for trajectory forecasting tasks. """
    
    def __init__(self, config: ModelConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
        
        # note: this could / should be part of the dataset generation
        # however, for autoregressive predictions, it is a pain to unravel these things
        # therefore, the model just adds them dynamically
        if config.positional_encoding is not None:
            self.register_buffer('positional_encoding', positional_encoding(
                config.positional_encoding.max_sequence_length, 
                config.positional_encoding.dim))
            self.append_positional_encodings_to_edge_features = config.positional_encoding.append_to_edge_features
            self.append_positional_encodings_to_node_features = config.positional_encoding.append_to_node_features
            self.append_positional_encodings_to_edge_to_trajectory_adjacency_features = config.positional_encoding.append_to_edge_to_trajectory_adjacency_features
        else:
            self.positional_encoding = None
            
        self.one_hot_node_encoding_dim = dataset.num_nodes if config.node_one_hot_encoding else None
        self.constant_node_features = config.constant_node_features
        self.constant_edge_features = config.constant_edge_features
    
    @typechecked       
    def _compute_positional_encoding(self, batch: TrajectoryComplex) -> TensorType['num_edge_to_trajectory_adjacency', 'positional_encoding_dim']:
        """ Computes the positional encoding for a batch. """
        offsets = [0] * batch.num_trajectories
        positions = []
        for idx in batch.edge_to_trajectory_adjacency[1].tolist():
            positions.append(offsets[idx])
            offsets[idx] +=1
        return self.positional_encoding[positions]
        
    @property
    def multiple_steps_predictions_differientable(self) -> bool:
        """ If predictions over more than one step are differentiable. By default, as this prediction is autoregressive, and uses argmax / sampling,
        this is False. """
        return False
    
    @property
    def num_appended_edge_features(self) -> int:
        if self.positional_encoding is not None and self.append_positional_encodings_to_edge_features:
            return self.positional_encoding.size(-1)
        else:
            return 0
    
    @property
    def num_appended_edge_to_trajectory_adjacency_features(self) -> int:
        if self.positional_encoding is not None and self.append_positional_encodings_to_edge_to_trajectory_adjacency_features:
            return self.positional_encoding.size(-1)
        else:
            return 0
        
    @property
    def num_appended_node_features(self) -> int:
        num = 0
        if self.positional_encoding is not None and self.append_positional_encodings_to_node_features:
            num += self.positional_encoding.size(-1)
        if self.one_hot_node_encoding_dim is not None:
            num += self.one_hot_node_encoding_dim
        return num   
    
    @property
    def receptive_field(self) -> int | None:
        # by default, model's have an infinite receptive field
        return None

    @typechecked
    def num_node_features(self, dataset: TrajectoryForecastingDataset) -> int:
        if self.constant_node_features:
            return 1
        else:
            return dataset.num_node_features + self.num_appended_node_features
    
    @typechecked
    def num_edge_features(self, dataset: TrajectoryForecastingDataset) -> int:
        if self.constant_edge_features:
            return 1
        else:
            return dataset.num_edge_features

    @typechecked       
    def _append_positional_encoding_to_node_features(self, batch: TrajectoryComplex, 
                                                     positional_encoding: TensorType['max_sequence_length', 'positional_encoding_dim'], 
                                                     inputs: List[TensorType | None]):
        """ Appends positional encodings to the node features (in-place)"""
        broadcasted = torch.zeros(batch.edge_complex.num_nodes, positional_encoding.size(-1), 
                                        dtype=positional_encoding.dtype, device=positional_encoding.device)
        node_idxs = batch.edge_complex.node_adjacency[batch.edge_to_trajectory_orientation, 
                                                        batch.edge_complex.edge_idxs_to_node_adjacency_idxs(batch.edge_to_trajectory_adjacency[0])]
        broadcasted[node_idxs] = positional_encoding

        # we are now missing the endpoint of each trajectory, as we only picked the starting point of each edge
        endpoint_edge_idxs, endpoint_orientation, endpoint_trajectory_idxs = batch.trajectory_endpoints 
        endpoint_positions = batch.trajectory_lengths[endpoint_trajectory_idxs]
        endpoint_node_idxs = batch.edge_complex.node_adjacency[1 - endpoint_orientation, batch.edge_complex.edge_idxs_to_node_adjacency_idxs(endpoint_edge_idxs)]
        broadcasted[endpoint_node_idxs] = self.positional_encoding[endpoint_positions]
        
        if inputs[0] is None:
            inputs[0] = broadcasted
        else:
            inputs[0] = torch.cat((inputs[0], broadcasted), dim=-1)
            
    @typechecked       
    def _append_positional_encoding_to_edge_features(self, batch: TrajectoryComplex, 
                                                     positional_encoding: TensorType['max_sequence_length', 'positional_encoding_dim'], 
                                                     inputs: List[TensorType | None]):
        """ Appends positional encodings to the edge features (in-place)"""
        broadcasted = torch.zeros(batch.edge_complex.num_edges, positional_encoding.size(-1), 
                                        dtype=positional_encoding.dtype, device=positional_encoding.device)
        broadcasted[batch.edge_to_trajectory_adjacency[0]] = positional_encoding
        if inputs[1] is None:
            inputs[1] = broadcasted
        else:
            inputs[1] = torch.cat((inputs[1], broadcasted), dim=-1)
    
    @typechecked       
    def _append_positional_encoding_to_edge_to_trajectory_features(self, batch: TrajectoryComplex,
                                                     positional_encoding: TensorType['max_sequence_length', 'positional_encoding_dim']) -> TrajectoryComplex:
        """ Appends positional encodings to the edge_to_trajectory_adjacency features (not in-place)"""
        if batch.edge_to_trajectory_adjacency_features is None:
            edge_to_trajectory_adjacency_features = positional_encoding
        else:
            edge_to_trajectory_adjacency_features = torch.cat(
                (batch.edge_to_trajectory_adjacency_features, positional_encoding), dim=-1)
        batch = attrs.evolve(batch, edge_to_trajectory_adjacency_features = edge_to_trajectory_adjacency_features)
        return batch
    
    @typechecked
    def _append_one_hot_encoding_to_node_features(self, batch: TrajectoryComplex, 
                                                  inputs: List[TensorType | None]):
        inputs[0] = torch.cat((inputs[0], torch.eye(batch.edge_complex.num_nodes, device=inputs[0].device, dtype=inputs[0].dtype)), dim=-1)
    
    @typechecked       
    def prepare_inputs(self, batch: TrajectoryComplex) -> Tuple[List[TensorType | None], TrajectoryComplex]:
        """ Prepares inputs, e.g. applies positional encodings. """
        inputs = [batch.edge_complex.node_features, batch.edge_complex.edge_features, batch.trajectory_features]
        
        if self.positional_encoding is not None:
            # append positional encodings to attribute matrices
            positional_encoding = self._compute_positional_encoding(batch)
            if self.append_positional_encodings_to_node_features:
                self._append_positional_encoding_to_node_features(batch, positional_encoding, inputs)
            if self.append_positional_encodings_to_edge_features:
                self._append_positional_encoding_to_edge_features(batch, positional_encoding, inputs)
            if self.append_positional_encodings_to_edge_to_trajectory_adjacency_features:
                batch = self._append_positional_encoding_to_edge_to_trajectory_features(batch, positional_encoding)
        
        if self.one_hot_node_encoding_dim:
            self._append_one_hot_encoding_to_node_features(batch, inputs)   

        if self.constant_node_features:
            inputs[0] = torch.ones((inputs[0].size(0), 1), dtype=inputs[0].dtype, device=inputs[0].device)
            inputs[0][batch.trajectory_node_idxs[0], 0] = 1 # one-hot for all nodes that are in trajectories

        if self.constant_edge_features:
            inputs[1] = torch.ones((inputs[1].size(0), 1), dtype=inputs[1].dtype, device=inputs[1].device)
            inputs[1][batch.edge_to_trajectory_adjacency[0], 0] = 1 # one-hot for all edges that are in trajectories

        return inputs, batch
      
    @typechecked         
    def sample_successors(self, batch: TrajectoryComplex, pred: TrajectoryForecastingPrediction) -> TensorType['num_trajectories']:
        """ Samples the successor indices from a data sample and the model predictions on it
        """
        # Default strategy: Sample the most likely node
        return pred.get_sampled_successor_candidate_idxs_by_max_score(by_order=0)
    
    @typechecked       
    def predict_multiple_steps(self, batch: TrajectoryComplex, steps: Iterable[int], context: str) -> List[TrajectoryForecastingPrediction | None]:
        """ Predicts multiple steps given a base trajectory in an autoregressive fashion. That is,
        the model is called `__call__` (i.e. `forward`) iteratively on the previously predicted trajectory.
        
        Parameters
        ----------
        batch : TrajectoryComplex
            the trajectory
        steps : Iterable[int]
            for which steps to predict
        context : str
            in which context ('train', 'val', 'test')

        Returns
        -------
        List[TrajectoryForecastingPrediction | None]
            the sequence of predictions
        """
        steps = set(steps)
        max_num_steps = max(steps) + 1
        result = []
        for step_idx in range(max_num_steps):
            prediction: TrajectoryForecastingPrediction = self(batch, context)
            with torch.no_grad():
                prediction.populate_successor_candidates_from_batch(batch)
                prediction.sampled_successor_candidate_idxs = self.sample_successors(batch, prediction)
                if step_idx < max_num_steps - 1:
                    batch = batch.advance(
                        prediction.successor_candidate_edge_idxs[prediction.sampled_successor_candidate_idxs],
                        prediction.successor_candidate_edge_orientations[prediction.sampled_successor_candidate_idxs],
                        prediction.successor_candidate_trajectory_idxs[prediction.sampled_successor_candidate_idxs],
                    )
            if step_idx in steps:
                result.append(prediction)
        return result
        
class TrajectoryForecastingModelOneShotNoGradient(TrajectoryForecasting):
    """ Base model for models that do not use gradient descent but instead have one-shot fashion fitting method."""
    
    def __init__(self, config: ModelConfig, dataset: TrajectoryForecastingDataset):
        super().__init__(config, dataset)
    
    def fit_batch(
        self, 
        batch: TrajectoryComplex,
        batch_idx: int):
        raise NotImplemented
    
    def __call__(self, batch: TrajectoryComplex, context: str, *args) -> TrajectoryForecastingPrediction:
        """ Predicts for a single batch. """
        raise NotImplemented
        
    @property
    def needs_gradient_descent_training(self) -> bool: return False
    
    
    
        