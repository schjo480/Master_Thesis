import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader
from dataset.trajctory_dataset import TrajectoryDataset, collate_fn
from .d3pm_diffusion import make_diffusion
from .d3pm_edge_encoder import Edge_Encoder
import yaml
from tqdm import tqdm
import logging
import os
import time
import wandb


class Graph_Diffusion_Model(nn.Module):
    def __init__(self, data_config, diffusion_config, model_config, train_config, test_config, wandb_config, model, nodes, edges):
        super(Graph_Diffusion_Model, self).__init__()
        
        # Data
        self.data_config = data_config
        self.train_data_path = self.data_config['train_data_path']
        self.test_data_path = self.data_config['test_data_path']
        self.history_len = self.data_config['history_len']
        self.future_len = self.data_config['future_len']
        self.num_classes = self.data_config['num_classes']
        self.nodes = nodes
        self.edges = edges
        self.edge_features = self.data_config['edge_features']
        self.num_edge_features = len(self.edge_features)
        
        # Diffusion
        self.diffusion_config = diffusion_config
        
        # Model
        self.model_config = model_config
        self.model = model # Edge_Encoder
        self.hidden_channels = self.model_config['hidden_channels']
        
        # Training
        self.train_config = train_config
        self.lr = self.train_config['lr']
        self.lr_decay = self.train_config['lr_decay']
        self.num_epochs = self.train_config['num_epochs']
        self.gradient_accumulation = self.train_config['gradient_accumulation']
        self.gradient_accumulation_steps = self.train_config['gradient_accumulation_steps']
        self.batch_size = self.train_config['batch_size'] if not self.gradient_accumulation else self.train_config['batch_size'] * self.gradient_accumulation_steps
        
        # Testing
        self.test_config = test_config
        self.test_batch_size = self.test_config['batch_size']
        
        # WandB
        self.wandb_config = wandb_config
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.wandb_config['project'],
            entity=self.wandb_config['entity'],
            notes=self.wandb_config['notes'],
            job_type=self.wandb_config['job_type'],
            config={**self.data_config, **self.diffusion_config, **self.model_config, **self.train_config}
        )
        self.exp_name = self.wandb_config['exp_name']
        wandb.run.name = self.exp_name

        # Logging
        self.dataset = self.data_config['dataset']
        self.model_dir = os.path.join("experiments", self.exp_name)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.dataset}_{log_name}"
        
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        log_dir = os.path.join(self.model_dir, log_name)
        file_handler = logging.FileHandler(log_dir)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.log.addHandler(file_handler)
        
        self.log_loss_every_steps = self.train_config['log_loss_every_steps']        
        
        # Build Components
        self._build_train_dataloader()
        self._build_test_dataloader()
        self._build_model()
        self._build_optimizer()
        
    def forward(self, x, t, y, train):
        # 1. Apply diffusion process to input trajectories
        # 2. Pass the output through a graph neural network
        # 3. Get condition c of history
        # 4. Concatenate output with with c and encoded timestep
        # 5. Pass through a feedforward MLP network to get logits of next edges
        pass
        
    def train(self):
        
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges)
        def model_fn(x, edge_index, t, edge_attr, condition=None): #takes in noised future trajectory (and diffusion timestep)
            return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
        
        for epoch in tqdm(range(self.num_epochs)):
            # Update learning rate via scheduler and log it
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"epoch": epoch, "learning_rate": current_lr})
            total_loss = 0
            if self.gradient_accumulation:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    # TODO: Get future edge indices in one hot fashion
                    # future_edge_indices_one_hot = data['future_one_hot_edges']
                    future_trajectory_indices = data["future_indices"]
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                
                    self.optimizer.zero_grad()
                    for i in range(min(self.gradient_accumulation_steps, history_edge_features.size(0))):
                        c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features[i].unsqueeze(0), mode='history')
                        loss = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=future_trajectory_indices[i].unsqueeze(0))
                        total_loss += loss / self.gradient_accumulation_steps
                        (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                    self.optimizer.step()
                    
            else:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    # TODO: Get future edge indices in one hot fashion
                    # future_edge_indices_one_hot = data['future_one_hot_edges']
                    future_trajectory_indices = data["future_indices"]
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                    
                    self.optimizer.zero_grad()
                    c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                    loss = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=future_trajectory_indices)
                    total_loss += loss
                    loss.backward()
                    self.optimizer.step()
            avg_loss = total_loss / len(self.train_data_loader)
            if epoch % self.log_loss_every_steps == 0:
                wandb.log({"epoch": epoch, "average_loss": avg_loss.item()})
                self.log.info(f"Epoch {epoch} Average Loss: {avg_loss.item()}")
                print("Epoch:", epoch)
                print("Loss:", avg_loss)
            
        self.save_model()
            
    def get_samples(self, load_model=False, model_path=None):
        if load_model:
            if model_path is None:
                raise ValueError("Model path must be provided to load model.")
            self.load_model(model_path)
        
        def model_fn(x, edge_index, t, edge_attr, condition=None):
            return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
        
        sample_list = []
        ground_truth_hist = []
        ground_truth_fut = []
        
        for data in tqdm(self.test_data_loader):
            history_edge_features = data["history_edge_features"]
            # TODO: Get history edge indices in one hot fashion
            # history_edge_indices_one_hot = data['history_one_hot_edges']
            history_edge_indices = data["history_indices"]
            # TODO: Get future edge indices in one hot fashion
            # future_edge_indices_one_hot = data['future_one_hot_edges']
            future_trajectory_indices = data["future_indices"]
            node_features = self.node_features
            edge_index = self.edge_tensor
            # Get condition
            c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
            
            samples = make_diffusion(self.diffusion_config, self.model_config, 
                                    num_edges=self.num_edges).p_sample_loop(model_fn=model_fn,
                                                                            shape=(self.test_batch_size, self.future_len, 1),
                                                                            node_features=node_features,
                                                                            edge_index=edge_index,
                                                                            edge_attr=history_edge_features,
                                                                            condition=c)
            sample_list.append(samples)
            ground_truth_hist.append(history_edge_indices)
            ground_truth_fut.append(future_trajectory_indices)
        # Original
        # samples_shape = (device_bs, *self.dataset.data_shape) = (bs, height, width, channels)
        # Ours
        # samples_shape = (bs, future_len, 1) --> OK
        
        # TODO: Compare the sample to ground truth, and plot it with the history!
        return sample_list, ground_truth_hist, ground_truth_fut
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'{self.exp_name}.pth'))
        self.log.info("Model saved!")
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.log.info("Model loaded!")
    
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        def lr_lambda(epoch):
            return 1.0 if epoch < 10000 else 0.9999 ** (epoch - 10000)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        print("> Optimizer and Scheduler built!")
        
        '''print("Parameters to optimize:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)'''
        
    def _build_train_dataloader(self):
        self.train_dataset = TrajectoryDataset(self.train_data_path, self.history_len, self.nodes, self.edges, self.future_len, self.edge_features)
        self.node_features = self.train_dataset.node_coordinates()
        self.edge_tensor = self.train_dataset.get_all_edges_tensor()
        # self.trajectory_edge_tensor = self.train_dataset.get_trajectory_edges_tensor(0)
        self.num_edges = self.train_dataset.get_n_edges()
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
                
        print("> Training Dataset loaded!")
    
    def _build_test_dataloader(self):
        self.test_dataset = TrajectoryDataset(self.test_data_path, self.history_len, self.nodes, self.edges, self.future_len, self.edge_features)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=True, collate_fn=collate_fn)
        print("> Test Dataset loaded!")
        
    def _build_model(self):
        self.model = self.model(self.model_config, self.history_len, self.future_len, self.num_classes,
                                nodes=self.nodes, edges=self.edges, node_features=self.node_features,
                                num_edges=self.num_edges, hidden_channels=self.hidden_channels, num_edge_features=self.num_edge_features)
        print("> Model built!")
        
        
'''model = Edge_Encoder
train_data_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic.h5'
test_data_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_test.h5'
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
num_layers=3

# Read the config file
with open('/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/config/D3PM/d3pm.yaml', 'r') as file:
    config = yaml.safe_load(file)

model = Graph_Diffusion_Model(config, model, train_data_path, test_data_path, nodes, edges, num_layers)
model.train()'''
'''sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=True, model_path='/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/synthetic_d3pm/synthetic_d3pm.pth')
print(sample_list)
print(ground_truth_hist)
print(ground_truth_fut)'''
