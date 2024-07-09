"""Train Benchmark models that are not diffusion based"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score
import yaml
from tqdm import tqdm
import logging
import os
import time
import wandb
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import numpy as np


from dataset.trajctory_dataset import TrajectoryDataset, collate_fn

class Benchmark_Models(nn.Module):
    def __init__(self, data_config, model_config, train_config, test_config, wandb_config, model):
        super(Benchmark_Models, self).__init__()
        # Data
        self.data_config = data_config
        self.train_data_path = self.data_config['train_data_path']
        self.test_data_path = self.data_config['test_data_path']
        self.history_len = self.data_config['history_len']
        self.future_len = self.data_config['future_len']
        self.num_classes = self.data_config['num_classes']
        self.edge_features = self.data_config['edge_features']
        
        # Model
        self.model_config = model_config
        self.model = model # Edge_Encoder
        self.hidden_channels = self.model_config['hidden_channels']
        self.condition_dim = self.model_config['condition_dim']
        self.num_layers = self.model_config['num_layers']
        
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
        self.eval_every_steps = self.test_config['eval_every_steps']
        
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
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.num_epochs):
            ground_truth_fut = []
            pred_fut = []
            self.model.train()
            total_loss = 0
            for batch in self.train_data_loader:
                history_edge_features = batch["history_edge_features"]
                last_history_edge = batch["history_indices"][:, -1]
                future_edge_indices = batch["future_indices"]
                future_edge_features = batch["future_edge_features"]
                future_edge_indices_one_hot = future_edge_features[:, :, 0]
                
                self.optimizer.zero_grad()
                
                logits, preds = self.model(history_edge_features, last_history_edge, self.line_graph)
                pred_fut.append(preds)
                ground_truth_fut.append(future_edge_indices_one_hot.unsqueeze(0))
                # Calculate the loss (cross-entropy)
                loss = F.cross_entropy(logits.view(-1, self.num_classes), future_edge_indices.view(-1))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_data_loader)
            f1_score = F1Score(task='binary', average='macro', num_classes=2)
            f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach(), torch.flatten(torch.cat(ground_truth_fut)).detach())
            # Logging
            if epoch % self.log_loss_every_steps == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
                wandb.log({"epoch": epoch + 1, "loss": avg_loss})
                wandb.log({"epoch": epoch + 1, "average_F1_score": f1_epoch.item()})
                self.log.info(f"Epoch {epoch} Average Loss: {avg_loss.item()}")
                print("Epoch:", epoch)
                print("Loss:", avg_loss.item())
                print("F1:", f1_epoch.item())
        
    def save_model(self):
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + '_' +  self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                 f'_hidden_dim_{self.hidden_channels}_condition_dim_{self.condition_dim}_layers_{self.num_layers}.pth')
        torch.save(self.model.state_dict(), save_path)
        self.log.info(f"Model saved at {save_path}!")
        
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
        print("Loading Training Dataset...")
        self.train_dataset = TrajectoryDataset(self.train_data_path, self.history_len, self.future_len, self.edge_features)
        self.G = self.train_dataset.graph
        self.nodes = self.G.nodes
        self.edges = self.G.edges(data=True)
        self.indexed_edges = self.train_dataset.edges
        self.num_edge_features = self.train_dataset.num_edge_features
        
        # Build the line graph and corresponding edge index
        edge_index = self._build_edge_index()
        self.line_graph = Data(edge_index=edge_index)
        
        self.edge_tensor = self.train_dataset.get_all_edges_tensor()
        self.num_edges = self.train_dataset.get_n_edges()
        
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
                
        print("> Training Dataset loaded!\n")
        
    def _build_edge_index(self):
        print("Building edge index for line graph...")
        edge_index = torch.tensor([[e[0], e[1]] for e in self.edges], dtype=torch.long).t().contiguous()
        edge_to_index = {tuple(e[:2]): e[2]['index'] for e in self.edges}
        line_graph_edges = []
        edge_list = edge_index.t().tolist()
        for i, (u1, v1) in tqdm(enumerate(edge_list), total=len(edge_list), desc="Processing edges"):
            for j, (u2, v2) in enumerate(edge_list):
                if i != j and (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
                    line_graph_edges.append((edge_to_index[(u1, v1)], edge_to_index[(u2, v2)]))

        # Create the edge index for the line graph
        edge_index = torch.tensor(line_graph_edges, dtype=torch.long).t().contiguous()
        print("> Edge index built!\n")
        
        return edge_index
    
    def _build_test_dataloader(self):
        self.test_dataset = TrajectoryDataset(self.test_data_path, self.history_len, self.future_len, self.edge_features)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=collate_fn)
        print("> Test Dataset loaded!")
        
    def _build_model(self):
        self.model = self.model(self.model_config, self.history_len, self.future_len, self.num_classes,
                                nodes=self.nodes, edges=self.edges,
                                num_edges=self.num_edges, hidden_channels=self.hidden_channels, num_edge_features=self.num_edge_features)
        print("> Model built!")