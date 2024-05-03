from seml.experiment import Experiment
from models.d3pm_graph_diffusion_model import Graph_Diffusion_Model
from models.d3pm_edge_encoder import Edge_Encoder


experiment = Experiment()


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
    
    edge_encoder = Edge_Encoder
    nodes = [(0, {'pos': (0.1, 0.65)}),
         (1, {'pos': (0.05, 0.05)}), 
         (2, {'pos': (0.2, 0.15)}), 
         (3, {'pos': (0.55, 0.05)}),
         (4, {'pos': (0.8, 0.05)}),
         (5, {'pos': (0.9, 0.1)}),
         (6, {'pos': (0.75, 0.15)}),
         (7, {'pos': (0.5, 0.2)}),
         (8, {'pos': (0.3, 0.3)}),
         (9, {'pos': (0.2, 0.3)}),
         (10, {'pos': (0.3, 0.4)}),
         (11, {'pos': (0.65, 0.35)}),
         (12, {'pos': (0.8, 0.5)}),
         (13, {'pos': (0.5, 0.5)}),
         (14, {'pos': (0.4, 0.65)}),
         (15, {'pos': (0.15, 0.6)}),
         (16, {'pos': (0.3, 0.7)}),
         (17, {'pos': (0.5, 0.7)}),
         (18, {'pos': (0.8, 0.8)}),
         (19, {'pos': (0.4, 0.8)}),
         (20, {'pos': (0.25, 0.85)}),
         (21, {'pos': (0.1, 0.9)}),
         (22, {'pos': (0.2, 0.95)}),
         (23, {'pos': (0.45, 0.9)}),
         (24, {'pos': (0.95, 0.95)}),
         (25, {'pos': (0.9, 0.4)}),
         (26, {'pos': (0.95, 0.05)})]
    edges = [(0, 21), (0, 1), (0, 15), (21, 22), (22, 20), (20, 23), (23, 24), (24, 18), (19, 14), (14, 15), (15, 16), (16, 20), (19, 20), (19, 17), (14, 17), (14, 16), (17, 18), (12, 18), (12, 13), (13, 14), (10, 14), (1, 15), (9, 15), (1, 9), (1, 2), (11, 12), (9, 10), (3, 7), (2, 3), (7, 8), (8, 9), (8, 10), (10, 11), (8, 11), (6, 11), (3, 4), (4, 5), (4, 6), (5, 6), (24, 25), (12, 25), (5, 25), (11, 25), (5, 26)]

    model = Graph_Diffusion_Model(data_config=data, wandb_config=wandb, diffusion_config=diffusion_config, model_config=model, train_config=training, test_config=testing, model=edge_encoder, nodes=nodes, edges=edges)
    if eval:
        sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=True, model_path='/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/synthetic_d3pm/synthetic_d3pm_seml.pth')
    else:    
        model.train()
    # The result will be stored in the MongoDB
    return {
        'sample_list': sample_list,
        'ground_truth_hist': ground_truth_hist,
        'ground_truth_fut': ground_truth_fut,
    }
