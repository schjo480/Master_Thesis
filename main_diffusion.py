# from seml.experiment import Experiment
import numpy as np
import seml
import torch
import os
from models.d3pm_graph_diffusion_model import Graph_Diffusion_Model
from models.d3pm_edge_encoder import Edge_Encoder
from models.d3pm_edge_encoder_mlp import Edge_Encoder_MLP
from models.d3pm_edge_encoder_residual import Edge_Encoder_Residual


experiment = seml.Experiment()

@experiment.automain
def main(data, wandb, diffusion_config, model, training, testing, eval):
    
    if model['name'] == 'edge_encoder':
        edge_encoder = Edge_Encoder
    elif model['name'] == 'edge_encoder_residual':
        edge_encoder = Edge_Encoder_Residual
    elif model['name'] == 'edge_encoder_mlp':
        edge_encoder = Edge_Encoder_MLP
    else:
        raise NotImplementedError(f"Model {model['name']} not implemented")
    model_config = model
    
    model = Graph_Diffusion_Model(data_config=data, wandb_config=wandb, diffusion_config=diffusion_config, model_config=model,
                                  train_config=training, test_config=testing, model=edge_encoder, pretrained=training['pretrained'])
    
    if eval:
        features = ''
        for feature in data['edge_features']:
            features += feature + '_'
        exp_name = wandb['exp_name']
        model_dir = os.path.join("experiments", exp_name)
        model_path = os.path.join(model_dir, 
                                 exp_name + '_' + model_config['name'] + features + '_' + f'_hist{data['history_len']}' + 
                                 f'_fut{data['future_len']}_' + model_config['transition_mat_type'] + '_' + diffusion_config['type'] + 
                                 f'_hidden_dim_{model_config['hidden_channels']}_time_dim_{str(model_config['time_embedding_dim'])}.pth')
        model.get_samples(load_model=True, model_path=model_path, task='predict', number_samples=1, save=True, test=True)
        # Multiple samples
        model.get_samples(load_model=True, model_path=model_path, task='predict', save=True, test=True)
    else:    
        model.train()
        # One-Shot samples
        model.get_samples(load_model=False, task='predict', number_samples=1, save=True)
        # Multiple samples
        model.get_samples(load_model=False, task='predict', save=True)
        