# TODO make an actual model registry of needed
from typing import Any

from data.trajectory_forecasting import TrajectoryForecastingDataset
from models.scnn import TrajectoryForecastingSCNN
from models.sequential import TrajectoryForecastingSequential
from models.markov_chain import TrajectoryMarkovChain
from models.base import BaseModel, TrajectoryForecasting
from models.gretel import Gretel
from models.scone import SCoNe
from models.cssrnn import CSSRNN
from data.base import BaseDataset
from config import ModelConfig

def get_trajectory_forecasting_model(config: ModelConfig, dataset: TrajectoryForecastingDataset) -> TrajectoryForecasting:
    if config.name == 'trajectory_forecasting_scnn':
        return TrajectoryForecastingSCNN(config, dataset.num_node_features, dataset.num_edge_features, dataset.num_trajectory_features)
    elif config.name == 'trajectory_forecasting_sequential':
        return TrajectoryForecastingSequential(config, dataset)
    elif config.name == 'trajectory_forecasting_markov_chain':
        return TrajectoryMarkovChain(config, dataset)
    elif config.name == 'trajectory_forecasting_gretel':
        return Gretel(config, dataset)
    elif config.name == 'scone':
        return SCoNe(config, dataset)
    elif config.name == 'cssrnn':
        return CSSRNN(config, dataset)
    else:
        raise ValueError(f'Unsupported trajectory forecasting model {config.name}')

def get_model(config: ModelConfig, dataset: Any) -> BaseModel:
    return get_trajectory_forecasting_model(config, dataset)