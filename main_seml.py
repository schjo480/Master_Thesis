# from seml.experiment import Experiment
import numpy as np
import seml
import torch
from models.d3pm_graph_diffusion_model import Graph_Diffusion_Model
from models.d3pm_edge_encoder import Edge_Encoder
from models.d3pm_edge_encoder_mlp import Edge_Encoder_MLP
from models.d3pm_edge_encoder_residual import Edge_Encoder_Residual


experiment = seml.Experiment()


'''@experiment.config
def config():
    name = "${config.model.name}_${config.data.dataset}"'''


'''@experiment.automain
def main(
    # Define your configuration parameters here
    dataset: 'config/data/tdrive.yaml',
    model: 'config/model/scone.yaml',
    seed: int,  # seml automatically assigns a random seed
):'''
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
    
    model = Graph_Diffusion_Model(data_config=data, wandb_config=wandb, diffusion_config=diffusion_config, model_config=model, train_config=training, test_config=testing, model=edge_encoder, pretrained=training['pretrained'])
    
    if eval:
        sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=True, model_path=testing['model_path'])
        return {
            'sample_list': sample_list,
            'ground_truth_hist': ground_truth_hist,
            'ground_truth_fut': ground_truth_fut,
        }
    else:    
        model.train()
        features = ''
        for feature in data['edge_features']:
            features += feature + '_'
        sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = model.get_samples(load_model=False, task='predict')
        torch.save(sample_list, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{wandb['exp_name']}' + '/' + f'{model_config['transition_mat_type']}' + '_' + f'{diffusion_config['type']}' + '/samples_raw_' + features + f'hist{data['history_len']}_fut_{data['future_len']}.pth')
        fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, sample_list, valid_ids, ground_truth_hist, ground_truth_fut = model.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, return_samples=True)
        torch.save(sample_list, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{wandb['exp_name']}' + '/' + f'{model_config['transition_mat_type']}' + '_' + f'{diffusion_config['type']}' + '/samples_' + features + f'hist{data['history_len']}_fut_{data['future_len']}.pth')
        torch.save(valid_ids, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{wandb['exp_name']}' + '/' + f'{model_config['transition_mat_type']}' + '_' + f'{diffusion_config['type']}' + '/valid_ids_' + features + f'hist{data['history_len']}_fut_{data['future_len']}.pth')
        torch.save(ground_truth_hist, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{wandb['exp_name']}' + '/' + f'{model_config['transition_mat_type']}' + '_' + f'{diffusion_config['type']}' + '/gt_hist_' + features + f'hist{data['history_len']}_fut_{data['future_len']}.pth')
        torch.save(ground_truth_fut, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{wandb['exp_name']}' + '/' + f'{model_config['transition_mat_type']}' + '_' + f'{diffusion_config['type']}' + '/gt_fut_' + features + f'hist{data['history_len']}_fut_{data['future_len']}.pth')
        
        #print("ground_truth_hist", ground_truth_hist)
        #print("ground_truth_fut", ground_truth_fut)
        #print("sample_list", sample_list)
        print("\n")
        print("Val F1 Score", f1)
        #wandb.log({"Val F1 Score, mult samples": f1})
        print("\n")
        print("Val Accuracy", acc)
        #wandb.log({"Val Accuracy, mult samples": acc})
        print("\n")
        print("Val TPR", tpr)
        #wandb.log({"Val TPR, mult samples": tpr})
        print("\n")
        valid_ids = [item if item is not None else testing['number_samples'] for sublist in valid_ids for item in sublist]
        print("Avg. number of samples until valid", sum(valid_ids) / len(valid_ids))
        print("\n")
        print("Average sample length", avg_sample_length)
        print("\n")
        print("Valid sample ratio", round(valid_sample_ratio, 3))
        print("\n")
        print("Val Future ratio", fut_ratio)
        print("\n")
        