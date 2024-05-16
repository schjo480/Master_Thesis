import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score
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
        self.lr_decay_parameter = self.train_config['lr_decay']
        self.learning_rate_warmup_steps = self.train_config['learning_rate_warmup_steps']
        self.num_epochs = self.train_config['num_epochs']
        self.gradient_accumulation = self.train_config['gradient_accumulation']
        self.gradient_accumulation_steps = self.train_config['gradient_accumulation_steps']
        self.batch_size = self.train_config['batch_size'] if not self.gradient_accumulation else self.train_config['batch_size'] * self.gradient_accumulation_steps
        
        # Testing
        self.test_config = test_config
        self.test_batch_size = self.test_config['batch_size']
        self.model_path = self.test_config['model_path']
        
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
            total_f1 = 0
            if self.gradient_accumulation:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    future_edge_indices_one_hot = data['future_one_hot_edges']
                    # Check if any entry in future_edge_indices_one_hot is not 0 or 1
                    if not torch.all((future_edge_indices_one_hot == 0) | (future_edge_indices_one_hot == 1)):
                        continue  # Skip this datapoint if the condition is not met
                    # future_trajectory_indices = data["future_indices"]
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                
                    self.optimizer.zero_grad()
                    for i in range(min(self.gradient_accumulation_steps, history_edge_features.size(0))):
                        c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features[i].unsqueeze(0), mode='history')
                        x_start = future_edge_indices_one_hot[i].unsqueeze(0)   # (batch_size, num_edges)
                        loss, preds = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=x_start)   # preds are of shape (num_edges,)
                        x_start = x_start.squeeze(0)    # (batch_size, num_edges) -> (num_edges,)
                        f1_score = F1Score(task='binary', average='micro', num_classes=self.num_classes)
                        total_f1 += f1_score(preds, x_start) / self.gradient_accumulation_steps
                        total_loss += loss / self.gradient_accumulation_steps
                        (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                    self.optimizer.step()
                    
            else:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    future_edge_indices_one_hot = data['future_one_hot_edges']
                    # future_trajectory_indices = data["future_indices"]
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                    
                    self.optimizer.zero_grad()
                    c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                    x_start = future_edge_indices_one_hot
                    loss, preds = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=x_start)
                    f1_score = F1Score(average='micro', num_classes=self.num_classes)
                    total_f1 += f1_score(preds, x_start)
                    total_loss += loss
                    loss.backward()
                    self.optimizer.step()
            avg_loss = total_loss / len(self.train_data_loader)
            avg_f1 = total_f1 / len(self.train_data_loader)
            if epoch % self.log_loss_every_steps == 0:
                wandb.log({"epoch": epoch, "average_loss": avg_loss.item()})
                wandb.log({"epoch": epoch, "average_F1_score": avg_f1.item()})
                self.log.info(f"Epoch {epoch} Average Loss: {avg_loss.item()}")
                print("Epoch:", epoch)
                print("Loss:", avg_loss)
                        
            if self.train_config['save_model'] and epoch % self.train_config['save_model_every_steps'] == 0:
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

            history_edge_indices = data["history_indices"]

            future_trajectory_indices = data["future_indices"]
            node_features = self.node_features
            edge_index = self.edge_tensor
            # Get condition
            c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
            
            samples = make_diffusion(self.diffusion_config, self.model_config, 
                                    num_edges=self.num_edges).p_sample_loop(model_fn=model_fn,
                                                                            shape=(self.test_batch_size, self.num_edges, 1), # before: (self.test_batch_size, self.future_len, 1)
                                                                            node_features=node_features,
                                                                            edge_index=edge_index,
                                                                            edge_attr=history_edge_features,
                                                                            condition=c)
            samples = torch.where(samples == 1)[1]
            sample_list.append(samples)
            ground_truth_hist.append(history_edge_indices)
            ground_truth_fut.append(future_trajectory_indices)

        return sample_list, ground_truth_hist, ground_truth_fut
    
    def visualize_predictions(self, samples, ground_truth_hist, ground_truth_fut, num_samples=5):
        """
        Visualize the predictions of the model along with ground truth data.

        :param samples: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        :param num_samples: Number of samples to visualize.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        samples, ground_truth_hist, ground_truth_fut = self.get_samples(load_model=True, model_path=self.test_config['model_path'])
        save_dir = f'{os.path.join(self.model_dir, f'{self.exp_name}', 'plots')}'
        os.makedirs(save_dir, exist_ok=True)
        
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        all_edges = {tuple(self.edges[idx]) for idx in range(len(self.edges))}
        G.add_edges_from(all_edges)
        
        pos = nx.get_node_attributes(G, 'pos')  # Retrieve node positions stored in node attributes

        for i in range(min(num_samples, len(samples))):
            plt.figure(figsize=(18, 8))            

            for plot_num, (title, edge_indices) in enumerate([
                ('Ground Truth History', ground_truth_hist[i]),
                ('Ground Truth Future', ground_truth_fut[i]),
                ('Predicted Future', samples[i])
            ]):
                plt.subplot(1, 3, plot_num + 1)
                plt.title(title)
                subgraph_edges = {tuple(self.edges[idx]) for idx in edge_indices if idx < len(self.edges)}

                # Draw all edges as muted gray
                nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=0.5, alpha=0.3, edge_color='gray')

                # Draw subgraph edges with specified color
                edge_color = 'gray' if plot_num == 0 else 'green' if plot_num == 1 else 'red'
                node_color = 'skyblue'# if plot_num == 0 else 'lightgreen' if plot_num == 1 else 'orange'
                nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
                nx.draw_networkx_edges(G, pos, edgelist=subgraph_edges, width=3, alpha=1.0, edge_color=edge_color)
                nx.draw_networkx_labels(G, pos, font_size=15)
            # Save plot
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
            plt.close()  # Close the figure to free memory
    
    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'{self.exp_name}.pth'))
        self.log.info(f"Model saved at {os.path.join(self.model_dir, f'{self.exp_name}.pth')}!")
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.log.info("Model loaded!")
    
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        def lr_lambda(epoch):
            if epoch < self.learning_rate_warmup_steps:
                return 1.0
            else:
                decay_lr = self.lr_decay_parameter ** (epoch - self.learning_rate_warmup_steps)
                return max(decay_lr, 2e-5 / self.lr)
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        print("> Optimizer and Scheduler built!")
        
        """print("Parameters to optimize:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)"""
        
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
        