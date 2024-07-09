# from seml.experiment import Experiment
import seml
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
    
    model = Graph_Diffusion_Model(data_config=data, wandb_config=wandb, diffusion_config=diffusion_config, model_config=model, train_config=training, test_config=testing, model=edge_encoder)
    
    if eval:
        sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=True, model_path=testing['model_path'])
        return {
            'sample_list': sample_list,
            'ground_truth_hist': ground_truth_hist,
            'ground_truth_fut': ground_truth_fut,
        }
    else:    
        model.train()
        model.get_samples(load_model=False, task='predict', number_samples=1, save=True)
            
