from dataclasses import dataclass, field
from omegaconf import MISSING

from typing import List, Any

@dataclass
class PositionalEncodingConfig:
    """ Config to compute positional encodings. """
    
    # dimension of the encoding
    dim: int = 32
    # maximal sequence length, as the encodings are pre-computed and cached
    max_sequence_length: int = 100
    
    # to which orders to append positional encodings to
    append_to_node_features: bool = True
    append_to_edge_features: bool = True
    append_to_edge_to_trajectory_adjacency_features: bool = True

@dataclass
class ModelConfig:
    """ Base configuration for models. """
    
    name: str = MISSING
    positional_encoding: PositionalEncodingConfig | None = None

    # Only for debugging: This should not be used in pratice, as it fixes the graph
    node_one_hot_encoding: bool = False

    # Only for debugging
    constant_node_features: bool = False
    constant_edge_features: bool = False

@dataclass
class MarkovChainConfig(ModelConfig):
    """ Configuration for markov chains. """
    
    name: str = 'trajectory_forecasting_markov_chain'
    
    # size of the history (how many nodes)
    order: int = 1
    # pseudo count of the prior
    pseudo_count: int = 0
    # caches probabilities before making predictions
    cached: bool = True
    # the horizont to which to make a single prediction of ever observing an instance
    complex_probabilities_horizon: int = 12
    # verbosity
    verbose: bool = True
    # remove cycles of length 2, akin to Gretel's non-backtracking random walk
    non_backtracking: bool = False

@dataclass
class SequentialConfig(ModelConfig):
    """ Configuration for a sequential model. """
    
    # Which layers the sequential model has, each is directly passed to `hydra.instanciate`
    layers: List[Any] = field(default_factory=list)

@dataclass
class GretelConfig(ModelConfig):
    """ Configuration for the Gretel model. """
    
    name: str = 'trajectory_forecasting_gretel'
    
    # Gretel only allows for a fixed number of observed instances (i.e. nodes)
    num_observations: int = 5
    # If the diffusion to compute pseudo coordinates is parametric
    # note that parametric diffusion does not do anything by how Gretel is formulated
    parametric_diffusion: bool = False
    # number of diffusion steps for pseudo coordiantes
    num_diffusion_steps: int = 5
    # for parametric diffusion, how big the hidden size is
    diffusion_hidden_dim: int = 32
    # If the random walk matrix should be non-backtracking
    non_backtracking_random_walk: bool = True
    # If not enough observations are made (i.e. less nodes observed than `num_observations`)
    # zeros are padded either left (start of sequence) or right (end of sequence)
    left_padding: bool = True

@dataclass
class SCoNeConfig(ModelConfig):
    """ Configuration for the SCoNe model"""
    hidden_dims: List[int] = field(default_factory=lambda: [16, 16, 16])
    activation: str = 'tanh' # default tanh is orientation equivariant
    
    # What the input edge features should be
    # possible are:
    # - 'trajectory' : Like in the ScoNE paper, only -1 or 1 along the edges of the trajectory
    # - 'features' : Just edge features (potentially with positional encodings, if given)
    # - 'both' : Appends the edge features to the -1, 1 indicator of SCoNe
    input_features: str = 'trajectory'
    

@dataclass
class CSSRNNConfig(ModelConfig):
    """ Configuration for the CSSRNN model. """
    
    # Size of the edge embeddings
    edge_embedding_size: int = 400
    # How to embed edges both for inputs to the LSTM and querying (prediction)
    # possible are:
    # - 'learned' : Fixed embeddings per edge. Does not allow for different graphs at evaluation than training
    # - 'linear' : Linear transformation of edge input features
    # - 'both' : Sums results of 'learned' and 'linear'
    edge_embedding: str = 'learned'
    
    # Hidden units in each LSTM layer
    hidden_dim: int = 400
    # Number of LSTM layers
    num_layers: int = 1
    dropout: float = 0.1
    bidirectional: bool = False