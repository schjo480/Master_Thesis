import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score
#from dataset.trajctory_dataset import TrajectoryDataset, collate_fn
#from .d3pm_diffusion import make_diffusion
#from .d3pm_edge_encoder import Edge_Encoder
import yaml
from tqdm import tqdm
import logging
import os
import time
import wandb

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
        self.time_embedding_dim = self.model_config['time_embedding_dim']
        self.condition_dim = self.model_config['condition_dim']
        self.num_layers = self.model_config['num_layers']
        self.class_weights = self.model_config['class_weights']
        
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
        """
        Trains the diffusion-based trajectory prediction model.

        This function performs the training of the diffusion-based trajectory prediction model. It iterates over the specified number of epochs and updates the model's parameters based on the training data. The training process includes forward propagation, loss calculation, gradient computation, and parameter updates.

        Returns:
            None
        """
        
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges, future_len=self.future_len)
        def model_fn(x, edge_index, t, edge_attr, condition=None):
            if self.model_config['name'] == 'edge_encoder':
                return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_residual':
                return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(t=t, edge_attr=edge_attr, condition=condition, mode='future')
                
        for epoch in tqdm(range(self.num_epochs)):
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"epoch": epoch, "learning_rate": current_lr})
            
            total_loss = 0
            ground_truth_fut = []
            pred_fut = []
            if self.gradient_accumulation:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    future_edge_indices_one_hot = data['future_one_hot_edges']
                    # Check if any entry in future_edge_indices_one_hot is not 0 or 1
                    if not torch.all((future_edge_indices_one_hot == 0) | (future_edge_indices_one_hot == 1)):
                        continue  # Skip this datapoint if the condition is not met
                    
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                    
                    self.optimizer.zero_grad()
                    for i in range(min(self.gradient_accumulation_steps, history_edge_features.size(0))):
                        # Calculate history condition c
                        if self.model_config['name'] == 'edge_encoder':
                            c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features[i].unsqueeze(0), mode='history')
                        elif self.model_config['name'] == 'edge_encoder_residual':
                            c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features[i].unsqueeze(0), mode='history')
                        elif self.model_config['name'] == 'edge_encoder_mlp':
                            c = self.model.forward(edge_attr=history_edge_features[i].unsqueeze(0), mode='history')
                        
                        x_start = future_edge_indices_one_hot[i].unsqueeze(0)   # (batch_size, num_edges)
                        # Get loss and predictions
                        loss, preds = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=x_start)   # preds are of shape (num_edges,)
                        
                        ground_truth_fut.append(x_start)
                        pred_fut.append(preds)
                        
                        total_loss += loss / self.gradient_accumulation_steps
                        (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                    self.optimizer.step()
                    
            else:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    future_edge_indices_one_hot = data['future_one_hot_edges']
                    # Check if any entry in future_edge_indices_one_hot is not 0 or 1
                    if not torch.all((future_edge_indices_one_hot == 0) | (future_edge_indices_one_hot == 1)):
                        continue
                    
                    batch_size = future_edge_indices_one_hot.size(0)
                    if self.model_config['name'] == 'edge_encoder_mlp':
                        if batch_size == self.batch_size:
                            future_edge_indices_one_hot = future_edge_indices_one_hot.view(self.batch_size, self.num_edges, 1)
                        else:
                            future_edge_indices_one_hot = future_edge_indices_one_hot.view(batch_size, self.num_edges, 1)
                            
                    node_features = self.node_features
                    edge_index = self.edge_tensor
                    
                    self.optimizer.zero_grad()
                    # Calculate history condition c
                    if self.model_config['name'] == 'edge_encoder':
                        c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                    elif self.model_config['name'] == 'edge_encoder_residual':
                        c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                    elif self.model_config['name'] == 'edge_encoder_mlp':
                        c = self.model.forward(edge_attr=history_edge_features, mode='history')
                    
                    x_start = future_edge_indices_one_hot
                    # Get loss and predictions
                    loss, preds = dif.training_losses(model_fn, node_features, edge_index, data, c, x_start=x_start)
                    
                    x_start = x_start.squeeze(-1)   # (bs, num_edges, 1) -> (bs, num_edges)
                    ground_truth_fut.append(x_start)
                    pred_fut.append(preds)
                    
                    total_loss += loss
                    loss.backward()
                    self.optimizer.step()
            
            self.scheduler.step()
            
            avg_loss = total_loss / len(self.train_data_loader)
            f1_score = F1Score(task='binary', average='macro', num_classes=2)
            f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach(), torch.flatten(torch.cat(ground_truth_fut)).detach())
            # Logging
            if epoch % self.log_loss_every_steps == 0:
                wandb.log({"epoch": epoch, "average_loss": avg_loss.item()})
                wandb.log({"epoch": epoch, "average_F1_score": f1_epoch.item()})
                self.log.info(f"Epoch {epoch} Average Loss: {avg_loss.item()}")
                print("Epoch:", epoch)
                print("Loss:", avg_loss.item())
                print("F1:", f1_epoch.item())
                        
            if self.train_config['save_model'] and (epoch + 1) % self.train_config['save_model_every_steps'] == 0:
                self.save_model()
            
    def get_samples(self, load_model=False, model_path=None, task='predict', number_samples=1):
        """
        Retrieves samples from the model.

        Args:
            load_model (bool, optional): Whether to load a pre-trained model. Defaults to False.
            model_path (str, optional): The path to the pre-trained model. Required if `load_model` is True.
            task (str, optional): The task to perform. Defaults to 'predict'. Other possible value: 'generate' to generate realistic trajectories

        Returns:
            tuple: A tuple containing three lists:
                - sample_list (list): A list of samples generated by the model.
                - ground_truth_hist (list): A list of ground truth history edge indices.
                - ground_truth_fut (list): A list of ground truth future trajectory indices.
        """
        
        if load_model:
            if model_path is None:
                raise ValueError("Model path must be provided to load model.")
            self.load_model(model_path)
        
        if self.test_config['number_samples'] is not None:
            number_samples = self.test_config['number_samples']
        
        def model_fn(x, edge_index, t, edge_attr, condition=None):
            if self.model_config['name'] == 'edge_encoder':
                return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_residual':
                return self.model.forward(x, edge_index, t, edge_attr, condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(t=t, edge_attr=edge_attr, condition=condition, mode='future')
        
        sample_list = []
        ground_truth_hist = []
        ground_truth_fut = []
        
        if task == 'predict':
            for data in tqdm(self.test_data_loader):
                history_edge_features = data["history_edge_features"]

                history_edge_indices = data["history_indices"]

                future_trajectory_indices = data["future_indices"]
                node_features = self.node_features
                edge_index = self.edge_tensor
                # Get condition
                if self.model_config['name'] == 'edge_encoder':
                    c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                elif self.model_config['name'] == 'edge_encoder_residual':
                    c = self.model.forward(x=node_features, edge_index=edge_index, edge_attr=history_edge_features, mode='history')
                elif self.model_config['name'] == 'edge_encoder_mlp':
                    c = self.model.forward(edge_attr=history_edge_features, mode='history')
                
                if number_samples > 1:
                    new_seed = torch.seed() + torch.randint(0, 100000, (1,)).item()
                    torch.manual_seed(new_seed)
                    sample_sublist = []
                    for _ in range(number_samples):
                        samples = make_diffusion(self.diffusion_config, self.model_config, 
                                                num_edges=self.num_edges).p_sample_loop(model_fn=model_fn,
                                                                                        shape=(self.test_batch_size, self.num_edges, 1), # before: (self.test_batch_size, self.future_len, 1)
                                                                                        node_features=node_features,
                                                                                        edge_index=edge_index,
                                                                                        edge_attr=history_edge_features,
                                                                                        condition=c)
                        samples = torch.where(samples == 1)[1]
                        sample_sublist.append(samples)
                    sample_list.append(sample_sublist)
                elif number_samples == 1:
                    samples = make_diffusion(self.diffusion_config, self.model_config, 
                                            num_edges=self.num_edges).p_sample_loop(model_fn=model_fn,
                                                                                    shape=(self.test_batch_size, self.num_edges, 1), # before: (self.test_batch_size, self.future_len, 1)
                                                                                    node_features=node_features,
                                                                                    edge_index=edge_index,
                                                                                    edge_attr=history_edge_features,
                                                                                    condition=c)
                    samples = torch.where(samples == 1)[1]
                    sample_list.append(samples)
                else:
                    raise ValueError("Number of samples must be greater than 0.")
                ground_truth_hist.append(history_edge_indices)
                ground_truth_fut.append(future_trajectory_indices)
            
            if number_samples == 1:
                fut_ratio, f1, avg_sample_length = self.eval(sample_list, ground_truth_hist, ground_truth_fut)
                wandb.log({"F1 Score": f1.item()})
                wandb.log({"Future ratio": fut_ratio})
                wandb.log({"Average sample length": avg_sample_length})
            return sample_list, ground_truth_hist, ground_truth_fut
        
        elif task == 'generate':
            # Generate realistic trajectories without condition
            # Edge encoder model needs to be able to funciton with no edge_attr and no condition
            # Add generate mode to p_logits, p_sample, and p_sample_loop
            return
        
        else:
            raise NotImplementedError(task)
    
    def visualize_sample_density(self, samples, ground_truth_hist, ground_truth_fut, number_plots=5, number_samples=10):
        """
        Visualize the density of the samples generated by the model.

        :param samples: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import numpy as np

        samples, ground_truth_hist, ground_truth_fut = self.get_samples(load_model=True, model_path=self.test_config['model_path'], number_samples=number_samples)
        save_dir = f'{os.path.join(self.model_dir, f'{self.exp_name}', 'plots')}'
        os.makedirs(save_dir, exist_ok=True)

        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        all_edges = {tuple(self.edges[idx]) for idx in range(len(self.edges))}
        G.add_edges_from(all_edges)

        pos = nx.get_node_attributes(G, 'pos')

        for i in range(min(number_plots, len(samples))):
            plt.figure(figsize=(18, 8))            

            for plot_num, (title, edge_indices) in enumerate([
                ('Ground Truth History', ground_truth_hist[i][0]),
                ('Ground Truth Future', ground_truth_fut[i][0]),
                ('Predicted Future', samples[i])
            ]):
                plt.subplot(1, 3, plot_num + 1)
                plt.title(title)

                # Draw all edges as muted gray
                nx.draw_networkx_edges(G, pos, edgelist=all_edges, width=0.5, alpha=0.3, edge_color='gray')

                # Draw subgraph edges with specified color
                edge_color = 'gray' if plot_num == 0 else 'green' if plot_num == 1 else 'red'
                node_color = 'skyblue'
                if plot_num == 2:
                    edge_counts = np.zeros(len(all_edges))
                    for sample in samples[i]:
                        for edge in sample:
                            edge_counts[edge] += 1
                    max_count = np.max(edge_counts)
                    edge_widths = edge_counts / max_count
                    
                    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
                    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='red', width=edge_widths*5, alpha=edge_widths/np.max(edge_widths))
                    nx.draw_networkx_labels(G, pos, font_size=15)
                else:
                    subgraph_edges = {tuple(self.edges[idx]) for idx in edge_indices if idx < len(self.edges)}
                    nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=500)
                    nx.draw_networkx_edges(G, pos, edgelist=subgraph_edges, width=3, alpha=1.0, edge_color=edge_color)
                    nx.draw_networkx_labels(G, pos, font_size=15)
            # Save plot
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
            plt.close()  # Close the figure to free memory
    
    def visualize_predictions(self, samples, ground_truth_hist, ground_truth_fut, number_plots=5):
        """
        Visualize the predictions of the model along with ground truth data.

        :param samples: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        :param number_plots: Number of samples to visualize.
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

        for i in range(min(number_plots, len(samples))):
            plt.figure(figsize=(18, 8))            

            for plot_num, (title, edge_indices) in enumerate([
                ('Ground Truth History', ground_truth_hist[i][0]),
                ('Ground Truth Future', ground_truth_fut[i][0]),
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
    
    def eval(self, sample_list, ground_truth_hist, ground_truth_fut):
        """
        Evaluate the model's performance.

        :param sample_list: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        """
        def calculate_fut_ratio(sample_list, ground_truth_fut):
            """
            Calculates the ratio of samples in `sample_list` that have at least one or two edges in common with the ground truth future trajectory.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                tuple: A tuple containing the ratios of samples that have at least one or two edges in common with the ground truth future trajectory.
            """
            count_1 = 0
            count_2 = 0
            total = len(sample_list)

            for i, sample in enumerate(sample_list):
                edges_count = sum(1 for edge in ground_truth_fut[i][0] if edge in sample)
                if edges_count >= 1:
                    count_1 += 1
                if edges_count >= 2:
                    count_2 += 1

            ratio_1 = count_1 / total
            ratio_2 = count_2 / total
            return ratio_1, ratio_2
        
        def calculate_sample_f1(sample_list, ground_truth_fut):
            """
            Calculates the F1 score for a given list of samples and ground truth futures.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth futures.

            Returns:
                float: The F1 score.

            """
            one_hot_samples = [torch.zeros(self.num_edges) for _ in range(len(sample_list))]
            one_hot_futures = [torch.zeros(self.num_edges) for _ in range(len(ground_truth_fut))]
            for i, one_hot_sample in enumerate(one_hot_samples):
                for edge_index, edge in enumerate(self.edges):
                    if edge_index in sample_list[i]:
                        one_hot_sample[self.edges.index(edge)] = 1
            for i, one_hot_fut in enumerate(one_hot_futures):
                for edge_index, edge in enumerate(self.edges):
                    if edge_index in ground_truth_fut[i]:
                        one_hot_fut[self.edges.index(edge)] = 1
            metric = F1Score(task='binary', average='macro', num_classes=2)
            f1 = metric(torch.cat(one_hot_samples), torch.cat(one_hot_futures))

            return f1
        
        def calculate_avg_sample_length(sample_list):
            """
            Calculate the average sample length.

            Args:
                sample_list (list): A list of samples.

            Returns:
                float: The average sample length.
            """
            return sum(len(sample) for sample in sample_list) / len(sample_list)
        
        fut_ratio = calculate_fut_ratio(sample_list, ground_truth_fut)
        f1 = calculate_sample_f1(sample_list, ground_truth_fut)
        avg_sample_length = calculate_avg_sample_length(sample_list)
        
        return fut_ratio, f1, avg_sample_length
    
    def save_model(self):
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name+
                                 f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}_condition_dim_{self.condition_dim}_layers_{self.num_layers}_weights_{self.class_weights[0]}.pth')
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
        self.train_dataset = TrajectoryDataset(self.train_data_path, self.history_len, self.nodes, self.edges, self.future_len, self.edge_features)
        self.node_features = self.train_dataset.node_coordinates()
        self.edge_tensor = self.train_dataset.get_all_edges_tensor()
        # self.trajectory_edge_tensor = self.train_dataset.get_trajectory_edges_tensor(0)
        self.num_edges = self.train_dataset.get_n_edges()
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
                
        print("> Training Dataset loaded!")
    
    def _build_test_dataloader(self):
        self.test_dataset = TrajectoryDataset(self.test_data_path, self.history_len, self.nodes, self.edges, self.future_len, self.edge_features)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=collate_fn)
        print("> Test Dataset loaded!")
        
    def _build_model(self):
        self.model = self.model(self.model_config, self.history_len, self.future_len, self.num_classes,
                                nodes=self.nodes, edges=self.edges, node_features=self.node_features,
                                num_edges=self.num_edges, hidden_channels=self.hidden_channels, num_edge_features=self.num_edge_features)
        print("> Model built!")
        
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import networkx as nx
import numpy as np

def load_new_format(new_file_path):
    paths = []
    from tqdm import tqdm

    with h5py.File(new_file_path, 'r') as new_hf:
        node_coordinates = new_hf['graph']['node_coordinates'][:]
        edges = new_hf['graph']['edges'][:]
        edge_coordinates = node_coordinates[edges]
        nodes = [(i, {'pos': tuple(pos)}) for i, pos in enumerate(node_coordinates)]
        
        
        # Convert edges to a list of tuples
        edges = [tuple(edge) for edge in edges]

        for i in tqdm(new_hf['trajectories'].keys()):
            path_group = new_hf['trajectories'][i]
            path = {attr: path_group[attr][()] for attr in path_group.keys()}
            if 'edge_orientation' in path:
                path['edge_orientations'] = path.pop('edge_orientation')
            paths.append(path)

    return paths, nodes, edges, edge_coordinates

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, nodes, edges, future_len, edge_features=None):
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        # self.trajectories = h5py.File(file_path, 'r')
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = load_new_format(file_path)
        
        '''self.nodes = nodes
        self.edges = edges'''
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __getitem__(self, idx):
        # trajectory_name = self.keys[idx]
        trajectory = self.trajectories[idx]
        edge_idxs = torch.tensor(trajectory['edge_idxs'][:], dtype=torch.long)
        edge_orientations = torch.tensor(trajectory['edge_orientations'][:], dtype=torch.long)
        
        # edge_coordinates_data = trajectory.get('coordinates', [])
        edge_coordinates_data = self.edge_coordinates[edge_idxs]

        if len(edge_coordinates_data) > 0:
            edge_coordinates_np = np.array(edge_coordinates_data)
            edge_coordinates = torch.tensor(edge_coordinates_np, dtype=torch.float64)
        else:
            edge_coordinates = torch.tensor([], dtype=torch.float64)

        # Reverse coordinates if orientation is -1
        edge_coordinates[edge_orientations == -1] = edge_coordinates[edge_orientations == -1][:, [1, 0]]
        
        # Calculate the required padding length
        total_len = self.history_len + self.future_len
        padding_length = max(total_len - len(edge_idxs), 0)
        
        # Pad edge indices and orientations
        edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)
        edge_orientations = torch.nn.functional.pad(edge_orientations, (0, padding_length), value=0)
        
        # Pad coordinates
        if padding_length > 0 and edge_coordinates.numel() > 0:
            zero_padding = torch.zeros((padding_length, 2, 2), dtype=torch.float)
            edge_coordinates = torch.cat([edge_coordinates, zero_padding], dim=0)
        
        # Split into history and future
        history_indices = edge_idxs[:self.history_len]
        future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]
        history_coordinates = edge_coordinates[:self.history_len] if edge_coordinates.numel() > 0 else None
        future_coordinates = edge_coordinates[self.history_len:self.history_len + self.future_len] if edge_coordinates.numel() > 0 else None
        
        history_edge_orientations = torch.zeros(self.get_n_edges())
        future_edge_orientations = torch.zeros(self.get_n_edges())

        for index, i in enumerate(history_indices):
            history_edge_orientations[i] = edge_orientations[index]
        
        for index, i in enumerate(future_indices):
            future_edge_orientations[i] = edge_orientations[index]

        # One-hot encoding of edge indices (ensure valid indices first)
        valid_history_mask = history_indices >= 0
        valid_future_mask = future_indices >= 0
        
        history_one_hot_edges = torch.nn.functional.one_hot(history_indices[valid_history_mask], num_classes=len(self.edges))
        future_one_hot_edges = torch.nn.functional.one_hot(future_indices[valid_future_mask], num_classes=len(self.edges))
        
        # Sum across the time dimension to count occurrences of each edge
        history_one_hot_edges = history_one_hot_edges.sum(dim=0)  # (num_edges,)
        future_one_hot_edges = future_one_hot_edges.sum(dim=0)  # (num_edges,)

        if 'edge_orientations' in self.edge_features:
            history_edge_features = torch.stack((history_one_hot_edges, history_edge_orientations), dim=1)
            future_edge_features = torch.stack((future_one_hot_edges, future_edge_orientations), dim=1)
        else:
            history_edge_features = history_one_hot_edges
            future_edge_features = future_one_hot_edges
        
        # Generate the tensor indicating nodes in history
        node_in_history = torch.zeros((len(self.nodes), 1), dtype=torch.float)
        history_edges = [self.edges[i] for i in history_indices if i >= 0]
        history_nodes = set(node for edge in history_edges for node in edge)
        for node in history_nodes:
            node_in_history[node] = 1
        
        return {
            "history_indices": history_indices,
            "future_indices": future_indices,
            "history_coordinates": history_coordinates,
            "future_coordinates": future_coordinates,
            "history_one_hot_edges": history_one_hot_edges,
            "future_one_hot_edges": future_one_hot_edges,
            "history_edge_orientations": history_edge_orientations,
            "future_edge_orientations": future_edge_orientations,
            "history_edge_features": history_edge_features,
            "future_edge_features": future_edge_features,
            "node_in_history": node_in_history
        }
        
    def __len__(self):
        return len(self.trajectories)

    '''def __del__(self):
        self.trajectories.close()'''

    def get_n_edges(self):
        return self.graph.number_of_edges()
    
    def node_coordinates(self):
        """
        Returns a tensor of shape [#nodes, 2] containing the coordinates of each node.
        """
        coords = [attr['pos'] for _, attr in self.nodes]  # List of tuples (x, y)
        coords_tensor = torch.tensor(coords, dtype=torch.float)  # Convert list to tensor
        return coords_tensor
    
    def get_all_edges_tensor(self):
        """
        Returns a tensor of shape [2, num_edges] where each column represents an edge
        and the two entries in each column represent the nodes connected by that edge.
        """
        edges = list(self.graph.edges())
        edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        return edge_tensor


def collate_fn(batch):
    # Extract elements for each sample and stack them, handling variable lengths
    history_indices = torch.stack([item['history_indices'] for item in batch])
    future_indices = torch.stack([item['future_indices'] for item in batch])
    
    history_one_hot_edges = torch.stack([item['history_one_hot_edges'] for item in batch])
    future_one_hot_edges = torch.stack([item['future_one_hot_edges'] for item in batch])

    # Coordinates
    history_coordinates = [item['history_coordinates'] for item in batch if item['history_coordinates'] is not None]
    future_coordinates = [item['future_coordinates'] for item in batch if item['future_coordinates'] is not None]
    
    history_edge_orientations = torch.stack([item['history_edge_orientations'] for item in batch])
    future_edge_orientations = torch.stack([item['future_edge_orientations'] for item in batch])
    
    history_edge_features = torch.stack([item['history_edge_features'] for item in batch])
    future_edge_features = torch.stack([item['future_edge_features'] for item in batch])
    
    history_one_hot_nodes = torch.stack([item['node_in_history'] for item in batch])
    
    # Stack coordinates if not empty
    if history_coordinates:
        history_coordinates = torch.stack(history_coordinates)
    if future_coordinates:
        future_coordinates = torch.stack(future_coordinates)

    return {
            "history_indices": history_indices,
            "future_indices": future_indices,
            "history_coordinates": history_coordinates,
            "future_coordinates": future_coordinates,
            "history_one_hot_edges": history_one_hot_edges,
            "future_one_hot_edges": future_one_hot_edges,
            "history_edge_orientations": history_edge_orientations,
            "future_edge_orientations": future_edge_orientations,
            "history_edge_features": history_edge_features,
            "future_edge_features": future_edge_features,
            "history_one_hot_nodes": history_one_hot_nodes
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    """
    Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Edge_Encoder(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, node_features, num_edges, hidden_channels, num_edge_features):
        super(Edge_Encoder, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.node_features = node_features
        self.num_node_features = self.node_features.shape[1]
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = 1000.
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads)
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads)
        '''self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_node_features, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads))
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads))
        '''
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels*self.num_heads, self.condition_dim)  # To encode history to c
        self.future_decoder = nn.Linear(self.hidden_channels, self.num_edges)  # To predict future edges
        self.adjust_to_class_shape = nn.Conv1d(in_channels=self.num_nodes, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x, edge_index, t=None, edge_attr=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        
        # Edge Embedding
        x = F.relu(self.conv1(x, edge_index, edge_attr.squeeze(0)))
        x = F.relu(self.conv2(x, edge_index, edge_attr.squeeze(0)))
        x = F.relu(self.conv3(x, edge_index, edge_attr.squeeze(0)))
        '''for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr.squeeze(0)))'''
        x = x.unsqueeze(0).repeat(edge_attr.size(0), 1, 1) # Reshape x to [batch_size, num_nodes, feature_size]
        
        if mode == 'history':
            c = self.history_encoder(x)
            
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time)
            t_emb = self.time_linear0(t_emb)
            # TODO: Delete first silu function!
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)
            t_emb = t_emb.unsqueeze(1).repeat(1, self.num_nodes, 1)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=2) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=2) # Concatenate with condition c
            x = F.relu(nn.Linear(x.size(2), self.hidden_channels)(x))
            
            logits = self.future_decoder(x) # (bs, num_nodes, num_edges)
            logits = self.adjust_to_class_shape(logits) # (bs, num_classes=2, num_edges)
            logits = logits.permute(0, 2, 1)  # (bs, num_edges, num_classes=2)
            # Unsqueeze to get the final shape 
            logits = logits.unsqueeze(2)    # (batch_size, num_edges, 1, num_classes=2)

            return logits

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    """
    Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Edge_Encoder_MLP(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, node_features, num_edges, hidden_channels, num_edge_features):
        super(Edge_Encoder_MLP, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.node_features = node_features
        self.num_node_features = self.node_features.shape[1]
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = 1000.
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_layers = self.config['num_layers']
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(self.num_edges, self.hidden_channels))
        for _ in range(1, self.num_layers):
            self.lin_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels, self.condition_dim)  # To encode history to c
        self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim,
                                        self.num_edges)  # To predict future edges
        self.adjust_to_class_shape = nn.Conv1d(in_channels=1, out_channels=self.num_classes, kernel_size=3, padding=1)

    def forward(self, t=None, edge_attr=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        
        # Edge Embedding
        edge_attr = edge_attr.float()
        if edge_attr.dim() > 2:
            edge_attr = edge_attr.squeeze(2)
            if edge_attr.dim() > 2:
                edge_attr = edge_attr.squeeze(2)
        
        x = edge_attr   # (bs, hidden_dim)
        for layer in self.lin_layers:
            x = F.relu(layer(x))

        if mode == 'history':
            c = self.history_encoder(x) # (bs, condition_dim)
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)   # (bs, time_embedding_dim)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=1) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=1) # Concatenate with condition c
            # x has shape (bs, hidden_channels + time_embedding_dim + condition_dim)
            
            logits = F.relu(self.future_decoder(x)) # (bs, num_edges)
            logits = logits.unsqueeze(1) # (bs, 1, num_edges)
            logits = self.adjust_to_class_shape(logits) # (bs, num_classes=2, num_edges)
            logits = logits.permute(0, 2, 1)  # (bs, num_edges, num_classes=2)
            # Unsqueeze to get the final shape 
            logits = logits.unsqueeze(2)    # (bs, num_edges, 1, num_classes=2)

            return logits
        

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
    """
    Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Edge_Encoder_Residual(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, node_features, num_edges, hidden_channels, num_edge_features):
        super(Edge_Encoder_Residual, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.node_features = node_features
        self.num_node_features = self.node_features.shape[1]
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = 1000.
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_node_features, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads))
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads))
        
        self.res_layer = nn.Linear(self.num_edges, self.hidden_channels * self.num_heads)
        # self.res_layer = nn.Linear(self.num_node_features, self.hidden_channels * self.num_heads)


        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels*self.num_heads, self.condition_dim)  # To encode history to c
        self.future_decoder = nn.Linear(self.hidden_channels, self.num_edges)  # To predict future edges
        self.adjust_to_class_shape = nn.Conv1d(in_channels=self.num_nodes, out_channels=self.num_classes, kernel_size=1)

    def forward(self, x, edge_index, t=None, edge_attr=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        
        # Edge Embedding
        edge_attr_res_layer = edge_attr.float()
        if edge_attr_res_layer.dim() > 2:
            edge_attr_res_layer = edge_attr_res_layer.squeeze(2)
            edge_attr_res_layer = edge_attr_res_layer.squeeze(2)
        x_res = x
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr.squeeze(0)))
            x = 0.001*x + F.relu(self.res_layer(edge_attr_res_layer))
            # x = x + F.relu(self.res_layer(x_res))
            
        x = x.unsqueeze(0).repeat(edge_attr.size(0), 1, 1) # Reshape x to [batch_size, num_nodes, hidden_dim]
        if mode == 'history':
            c = self.history_encoder(x)
            
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time)
            t_emb = self.time_linear0(t_emb)
            # TODO: Delete first silu function!
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)
            t_emb = t_emb.unsqueeze(1).repeat(1, self.num_nodes, 1)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=2) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=2) # Concatenate with condition c
            x = F.relu(nn.Linear(x.size(2), self.hidden_channels)(x))
            
            logits = self.future_decoder(x) # (bs, num_nodes, num_edges)
            logits = self.adjust_to_class_shape(logits) # (bs, num_classes=2, num_edges)
            logits = logits.permute(0, 2, 1)  # (bs, num_edges, num_classes=2)
            # Unsqueeze to get the final shape 
            logits = logits.unsqueeze(2)    # (batch_size, num_edges, 1, num_classes=2)

            return logits

# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Diffusion for discrete state spaces."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_diffusion(diffusion_config, model_config, num_edges, future_len):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(diffusion_config),
        model_prediction=model_config['model_prediction'],
        model_output=model_config['model_output'],
        transition_mat_type=model_config['transition_mat_type'],
        transition_bands=model_config['transition_bands'],
        loss_type=model_config['loss_type'],
        hybrid_coeff=model_config['hybrid_coeff'],
        num_edges=num_edges,
        class_weights=model_config['class_weights'],
        model_name=model_config['name'],
        future_len=future_len
)


def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec['start'], spec['stop'], spec['num_timesteps'])
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = torch.linspace(0, 1, spec['num_timesteps'] + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2)
        betas = torch.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], torch.tensor(0.999))
        return betas
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec['num_timesteps'], 1, spec['num_timesteps'])
    else:
        raise NotImplementedError(spec['type'])


class CategoricalDiffusion:
    """Discrete state space diffusion process.

    Time convention: noisy data is labeled x_0, ..., x_{T-1}, and original data
    is labeled x_start (or x_{-1}). This convention differs from the papers,
    which use x_1, ..., x_T for noisy data and x_0 for original data.
    """

    def __init__(self, *, betas, model_prediction, model_output,
               transition_mat_type, transition_bands, loss_type, hybrid_coeff,
               num_edges, class_weights, torch_dtype=torch.float32, model_name=None, future_len=None):

        self.model_prediction = model_prediction  # *x_start*, xprev
        self.model_output = model_output  # logits or *logistic_pars*
        self.loss_type = loss_type  # kl, *hybrid*, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff
        self.class_weights = torch.tensor(class_weights)
        self.torch_dtype = torch_dtype
        self.model_name = model_name

        # Data \in {0, ..., num_edges-1}
        self.num_classes = 2 # 0 or 1
        self.num_edges = num_edges
        self.future_len = future_len
        self.class_probs = torch.tensor([1 - self.future_len / self.num_edges, self.future_len / self.num_edges], dtype=torch.float64)
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        if not isinstance(betas, torch.Tensor):
            raise ValueError('expected betas to be a torch tensor')
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')

        # Computations here in float64 for accuracy
        self.betas = betas.to(dtype=torch.float64)
        self.num_timesteps, = betas.shape

        # Construct transition matrices for q(x_t|x_{t-1})
        # NOTE: t goes from {0, ..., T-1}
        if self.transition_mat_type == 'uniform':
            q_one_step_mats = [self._get_transition_mat(t) 
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'gaussian':
            q_one_step_mats = [self._get_gaussian_transition_mat(t)
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'absorbing':
            q_one_step_mats = [self._get_absorbing_transition_mat(t)
                            for t in range(0, self.num_timesteps)]
        elif self.transition_mat_type == 'marginal_prior':
            q_one_step_mats = [self._get_prior_distribution_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing' "
                f", but is {self.transition_mat_type}"
                )

        self.q_onestep_mats = torch.stack(q_one_step_mats, axis=0)
        assert self.q_onestep_mats.shape == (self.num_timesteps,
                                            self.num_classes,
                                            self.num_classes)

        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                    dims=[[1], [0]])
            q_mats.append(q_mat_t)
        self.q_mats = torch.stack(q_mats, axis=0)
        assert self.q_mats.shape == (self.num_timesteps, self.num_classes,
                                    self.num_classes), self.q_mats.shape

        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = torch.transpose(self.q_onestep_mats, dim0=1, dim1=2)
        del self.q_onestep_mats

    def _get_full_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Contrary to the band diagonal version, this method constructs a transition
        matrix with uniform probability to all other states.

        Args:
            t: timestep. integer scalar.

        Returns:
            Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        # Create a matrix filled with beta_t/num_classes
        mat = torch.full((self.num_classes, self.num_classes), 
                            fill_value=beta_t / float(self.num_classes),
                            dtype=torch.float64)

        # Create a diagonal matrix with values to be set on the diagonal of mat
        diag_val = 1. - beta_t * (self.num_classes - 1.) / self.num_classes
        diag_matrix = torch.diag(torch.full((self.num_classes,), diag_val, dtype=torch.float64))

        # Set the diagonal values
        mat.fill_diagonal_(diag_val)

        return mat

    def _get_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition
        matrix Q with
        Q_{ij} = beta_t / num_classes       if |i-j| <= self.transition_bands
                1 - \sum_{l \neq i} Q_{il} if i==j.
                0                          else.

        Args:
        t: timestep. integer scalar (or numpy array?)

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        if self.transition_bands is None:
            return self._get_full_transition_mat(t)
        # Assumes num_off_diags < num_classes
        beta_t = self.betas[t]
        
        mat = torch.zeros((self.num_classes, self.num_classes),
                        dtype=torch.float64)
        off_diag = torch.full((self.num_classes - 1,), fill_value=beta_t / float(self.num_classes), dtype=torch.float64)

        for k in range(1, self.transition_bands + 1):
            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)
            off_diag = off_diag[:-1]

        # Add diagonal values such that rows sum to one
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag(diag)
        
        return mat

    def _get_gaussian_transition_mat(self, t):
        r"""Computes transition matrix for q(x_t|x_{t-1}).

        This method constructs a transition matrix Q with
        decaying entries as a function of how far off diagonal the entry is.
        Normalization option 1:
        Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                    1 - \sum_{l \neq i} Q_{il}  if i==j.
                    0                          else.

        Normalization option 2:
        tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= self.transition_bands
                            0                        else.

        Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

        Args:
            t: timestep. integer scalar (or numpy array?)

        Returns:
            Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        transition_bands = self.transition_bands if self.transition_bands else self.num_classes - 1

        beta_t = self.betas[t]

        mat = torch.zeros((self.num_classes, self.num_classes),
                        dtype=torch.float64)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = torch.linspace(torch.tensor(0.), torch.tensor(self.num_classes-1), self.num_classes, dtype=torch.float64)
        values = values * 2./ (self.num_classes - 1.)
        values = values[:transition_bands+1]
        values = -values * values / beta_t
        
        # To reverse the tensor 'values' starting from the second element
        reversed_values = values[1:].flip(dims=[0])
        # Concatenating the reversed values with the original values
        values = torch.cat([reversed_values, values], dim=0)
        values = F.softmax(values, dim=0)
        values = values[transition_bands:]
        
        for k in range(1, transition_bands + 1):
            off_diag = torch.full((self.num_classes - k,), values[k], dtype=torch.float64)

            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag_embed(diag)

        return mat

    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_classes//2.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]

        diag = torch.full((self.num_classes,), 1. - beta_t, dtype=torch.float64)
        mat = torch.diag(diag)

        # Add beta_t to the num_classes/2-th column for the absorbing state
        mat[:, self.num_classes // 2] += beta_t

        return mat
    
    def _get_prior_distribution_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).
        Use cosine schedule for these transition matrices.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]
        mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    mat[i, j] = beta_t * self.class_probs[j]
                else:
                    mat[i, j] = 1 - beta_t + beta_t * self.class_probs[j]
        
        return mat

    def _at(self, a, t, x):
        """
        Extract coefficients at specified timesteps t and conditioning data x in PyTorch.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one-hot representation, but have integer
            values representing the class values. --> NOT A LOT NEEDS TO CHANGE, MY CLASS VALUES ARE SIMPLY 0 AND 1

        Returns:
        a[t, x]: torch.Tensor: PyTorch tensor.
        """
        ### Original ###
        # x.shape = (bs, height, width, channels)
        # t_broadcast_shape = (bs, 1, 1, 1)
        # a.shape = (num_timesteps, num_pixel_vals, num_pixel_vals)
        # out.shape = (bs, height, width, channels, num_pixel_vals)
        # out[i, j, k, l, m] = a[t[i, j, k, l], x[i, j, k, l], m]
        
        ### New ###
        # x.shape = (bs, num_edges, channels=1) 
        # t_broadcast_shape = (bs, 1, 1)
        # a.shape = (num_timesteps, num_classes, num_classes) 
        # out.shape = (bs, num_edges, channels, num_classes) 
        
        # Convert `a` to the desired dtype if not already
        a = a.type(self.torch_dtype)

        # Prepare t for broadcasting by adding necessary singleton dimensions
        t_broadcast = t.view(-1, *((1,) * (x.ndim - 1)))

        # Advanced indexing in PyTorch to select elements
        return a[t_broadcast, x]

    def _at_onehot(self, a, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: torch.tensor: Jax array. output of dot(x, a[t], axis=[[-1], [1]]).
            shape = (bs, num_edges, channels=1, num_classes)
        """
        a = a.type(self.torch_dtype)

        ### New ###
        # t.shape = (bs)
        # x.shape = (bs, num_edges, channels=1, num_classes)
        # a[t].hape = (bs, num_classes, num_classes)
        # out.shape = (bs, num_edges, channels=1, num_classes)

        a_t = a[t]
        out = torch.einsum('bijc,bjk->bik', x, a_t) # Multiply last dimension of x with last 2 dimensions of a_t
        out = out.unsqueeze(2) # Add channel dimension
        
        return out

    def q_probs(self, x_start, t):
        """Compute probabilities of q(x_t | x_start).

        Args:
        x_start: torch.tensor: tensor of shape (bs, ...) of int32 or int64 type.
            Should not be of one hot representation, but have integer values
            representing the class values.
        t: torch.tensor: torch tensor of shape (bs,).

        Returns:
        probs: torch.tensor: shape (bs, x_start.shape[1:],
                                                num_classes).
        """
        return self._at(self.q_mats, t, x_start)

    def q_sample(self, x_start, t, noise):
        """
        Sample from q(x_t | x_start) (i.e. add noise to the data) using Gumbel softmax trick.

        Args:
        x_start: torch.tensor: original clean data, in integer form (not onehot).
            shape = (bs, ...).
        t: torch.tensor: timestep of the diffusion process, shape (bs,).
        noise: torch.tensor: uniform noise on [0, 1) used to sample noisy data.
            shape should match (*x_start.shape, num_classes).

        Returns:
        sample: torch.tensor: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps)

        # To avoid numerical issues, clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)
    
    def _get_logits_from_logistic_pars(self, loc, log_scale):
        """
        Computes logits for an underlying logistic distribution.

        Args:
        loc: torch.tensor: location parameter of logistic distribution.
        log_scale: torch.tensor: log scale parameter of logistic distribution.

        Returns:
        logits: torch.tensor: logits corresponding to logistic distribution
        """
        loc = loc.unsqueeze(-1)
        log_scale = log_scale.unsqueeze(-1)

        # Adjust the scale such that if it's zero, the probabilities have a scale
        # that is neither too wide nor too narrow.
        inv_scale = torch.exp(- (log_scale - 2.))

        bin_width = 2. / (self.num_classes - 1.)
        bin_centers = torch.linspace(-1., 1., self.num_classes)

        bin_centers = bin_centers.unsqueeze(0)  # Add batch dimension
        bin_centers = bin_centers - loc

        log_cdf_min = -F.softplus(-inv_scale * (bin_centers - 0.5 * bin_width))
        log_cdf_plus = -F.softplus(-inv_scale * (bin_centers + 0.5 * bin_width))

        logits = torch.log(torch.exp(log_cdf_plus) - torch.exp(log_cdf_min) + self.eps)

        return logits

    def q_posterior_logits(self, x_start, x_t, t, x_start_logits):
        """Compute logits of q(x_{t-1} | x_t, x_start) in PyTorch."""
        
        if (self.model_name == 'edge_encoder_mlp') & (x_t.dim() == 2):
            x_t = x_t.unsqueeze(-1)  # [batch_size, num_edges] --> [batch_size, num_edges, channels=1]
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)

        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t)
        if x_start_logits:
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Adds new dimensions: [batch_size, 1, 1, 1]
        t_broadcast = t_broadcast.expand(-1, tzero_logits.size(1), 1, tzero_logits.size(-1))   # tzero_logits.size(1) = num_edges, tzero_logits.size(-1) = num_classes
        
        return torch.where(t_broadcast == 0, tzero_logits, out) # (bs, num_edges, channels=1, num_classes)

    def p_logits(self, model_fn, x, t, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Compute logits of p(x_{t-1} | x_t) in PyTorch.

        Args:
            model_fn (function): The model function that takes input `x` and `t` and returns the model output.
            x (torch.Tensor): The input tensor of shape (batch_size, input_size) representing the noised input at time t.
            t (torch.Tensor): The time tensor of shape (batch_size,) representing the time step.

        Returns:
            tuple: A tuple containing two tensors:
                - model_logits (torch.Tensor): The logits of p(x_{t-1} | x_t) of shape (batch_size, input_size, num_classes).
                - pred_x_start_logits (torch.Tensor): The logits of p(x_{t-1} | x_start) of shape (batch_size, input_size, num_classes).
        """
        assert t.shape == (x.shape[0],)
        model_output = model_fn(node_features, edge_index, t, edge_attr, condition=condition)

        if self.model_output == 'logits':
            model_logits = model_output
        elif self.model_output == 'logistic_pars':
            loc, log_scale = model_output
            model_logits = self._get_logits_from_logistic_pars(loc, log_scale)
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits
            t_broadcast = t.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Adds new dimensions: [batch_size, 1, 1, 1]
            t_broadcast = t_broadcast.expand(-1, pred_x_start_logits.size(1), 1, pred_x_start_logits.size(-1))   # pred_x_start_logits.size(1) = num_edges, pred_x_start_logits.size(-1) = num_classes

            model_logits = torch.where(t_broadcast == 0, pred_x_start_logits,
                                       self.q_posterior_logits(pred_x_start_logits, x, t, x_start_logits=True))
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)
        
        assert (model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits
    
    # === Sampling ===

    def p_sample(self, model_fn, x, t, noise, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        # Get model logits
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)
        assert noise.shape == model_logits.shape, noise.shape

        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *([1] * (len(x.shape) - 1)))
        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))

        sample = torch.argmax(model_logits + nonzero_mask * gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        return sample, F.softmax(pred_x_start_logits, dim=-1)

    def p_sample_loop(self, model_fn, shape, num_timesteps=None, return_x_init=False, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Ancestral sampling."""
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.transition_mat_type in ['gaussian', 'uniform']:
            x_init = torch.randint(0, self.num_classes, size=shape, device=device)
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(shape, fill_value=self.num_classes // 2, dtype=torch.int32, device=device)
        else:
            raise ValueError(f"Invalid transition_mat_type {self.transition_mat_type}")

        x = x_init.clone()
        edge_attr = x_init.unsqueeze(-1).type(torch.float32)
        
        for i in range(num_timesteps):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i, dtype=torch.long, device=device)
            noise = torch.rand(x.shape + (self.num_classes,), device=device, dtype=torch.float32)
            x, _ = self.p_sample(model_fn=model_fn, x=x, t=t, noise=noise, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)
            edge_attr = x.unsqueeze(-1).type(torch.float32)

        if return_x_init:
            return x_init, x
        else:
            return x

  # === Log likelihood / loss calculation ===

    def vb_terms_bpd(self, model_fn, *, x_start, x_t, t, node_features=None, edge_index=None, edge_attr=None, condition=None):
        """Calculate specified terms of the variational bound.

        Args:
        model_fn: the denoising network
        x_start: original clean data
        x_t: noisy data
        t: timestep of the noisy data (and the corresponding term of the bound
            to return)

        Returns:
        a pair `(kl, pred_start_logits)`, where `kl` are the requested bound terms
        (specified by `t`), and `pred_x_start_logits` is logits of
        the denoised image.
        """
        true_logits = self.q_posterior_logits(x_start, x_t, t, x_start_logits=False)
        model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr, condition=condition)

        kl = categorical_kl_logits(input_logits=model_logits, target_logits=true_logits)
        assert kl.shape == x_start.shape
        kl = kl / torch.log(torch.tensor(2.0))

        decoder_nll = F.cross_entropy(model_logits, x_start, weight=self.class_weights, reduction='mean')
        # decoder_nll = -categorical_log_likelihood(x_start, model_logits)
        #assert decoder_nll.shape == x_start.shape

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_start) || p(x_{t-1}|x_t))
        assert kl.shape == decoder_nll.shape == t.shape == (x_start.shape[0],)
        result = torch.where(t == 0, decoder_nll, kl)
        return result, pred_x_start_logits

    def prior_bpd(self, x_start):
        """KL(q(x_{T-1}|x_start)|| U(x_{T-1}|0, num_edges-1))."""
        q_probs = self.q_probs(
            x_start=x_start,
            t=torch.full((x_start.shape[0],), self.num_timesteps - 1, dtype=torch.long))

        if self.transition_mat_type in ['gaussian', 'uniform']:
            # Stationary distribution is a uniform distribution over all pixel values.
            prior_probs = torch.ones_like(q_probs) / self.num_classes
        elif self.transition_mat_type == 'absorbing':
            absorbing_int = torch.full(x_start.shape[:-1], self.num_classes // 2, dtype=torch.int32)
            prior_probs = F.one_hot(absorbing_int, num_classes=self.num_classes).to(dtype=self.torch_dtype)
        else:
            raise ValueError("Invalid transition_mat_type")


        assert prior_probs.shape == q_probs.shape

        kl_prior = categorical_kl_probs(
            q_probs, prior_probs)
        assert kl_prior.shape == x_start.shape
        return meanflat(kl_prior) / torch.log(torch.tensor(2.0))
        
    def cross_entropy_x_start(self, x_start, pred_x_start_logits, class_weights):
        """Calculate binary weighted cross entropy between x_start and predicted x_start logits.

        Args:
            x_start (torch.Tensor): original clean data, expected binary labels (0 or 1)
            pred_x_start_logits (torch.Tensor): logits as predicted by the model
            class_weights (torch.Tensor): tensor with weights for class 0 and class 1

        Returns:
            torch.Tensor: scalar tensor representing the mean binary weighted cross entropy loss.
        """
        
        # Calculate binary cross-entropy with logits
        pred_x_start_logits = pred_x_start_logits.squeeze(2) # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
        pred_x_start_logits = pred_x_start_logits.permute(0, 2, 1) # (bs, num_edges, classes) -> (bs, classes, num_edges)
        if x_start.dim() == 3:
            x_start = x_start.squeeze(2) # (bs, num_edges, channels) -> (bs, num_edges)
        ce = F.cross_entropy(pred_x_start_logits, x_start, weight=class_weights, reduction='mean')

        return ce

    def training_losses(self, model_fn, node_features=None, edge_index=None, data=None, condition=None, *, x_start):
        """Training loss calculation."""
        # Add noise to data
        if self.model_name != 'edge_encoder_mlp':
            x_start = x_start.unsqueeze(-1)  # [batch_size, num_edges] --> [batch_size, num_edges, channels=1]
        noise = torch.rand(x_start.shape + (self.num_classes,), dtype=torch.float32)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],))

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        if (self.model_name == 'edge_encoder_mlp') & (x_t.dim() == 2):
            x_t = x_t.unsqueeze(-1)  # [batch_size, num_edges] --> [batch_size, num_edges, channels=1]
        
        edge_attr_t = x_t.unsqueeze(-1).type(torch.float32)

        # Calculate the loss
        if self.loss_type == 'kl':
            losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t,
                                                               node_features=node_features, edge_index=edge_index, edge_attr=edge_attr_t, condition=condition)
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            # NOTE: Currently only works for batch size of 1
            pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
            pred = pred_x_start_logits.argmax(dim=1)                # (num_edges, classes) -> (num_edges,)
            
            return losses, pred
            
        elif self.loss_type == 'cross_entropy_x_start':
            
            _, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, node_features=node_features, edge_index=edge_index, edge_attr=edge_attr_t, condition=condition)
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits, class_weights=self.class_weights)
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            
            if (self.model_name == 'edge_encoder') | (self.model_name == 'edge_encoder_residual'):
                # NOTE: Currently only works for batch size of 1
                pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
                pred = pred_x_start_logits.argmax(dim=1)    # (num_edges, classes) -> (num_edges,)
            elif self.model_name == 'edge_encoder_mlp':
                pred = pred_x_start_logits.argmax(dim=2)
            
            return losses, pred
            
        elif self.loss_type == 'hybrid':
            vb_losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t,
                                                               node_features=node_features, edge_index=edge_index, edge_attr=edge_attr_t, condition=condition)
            ce_losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits, class_weights=self.class_weights)
            losses = vb_losses + self.hybrid_coeff * ce_losses
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            # NOTE: Currently only works for batch size of 1
            pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
            pred = pred_x_start_logits.argmax(dim=1)    # (num_edges, classes) -> (num_edges,)
            
            return losses, pred
            
        else:
            raise NotImplementedError(self.loss_type)

        return losses

    def calc_bpd_loop(self, model_fn, *, x_start, rng):
        """Calculate variational bound (loop over all timesteps and sum)."""
        batch_size = x_start.shape[0]
        total_vb = torch.zeros(batch_size)

        for t in range(self.num_timesteps):
            noise = torch.rand(x_start.shape + (self.num_classes,), dtype=torch.float32)
            x_t = self.q_sample(x_start=x_start, t=torch.full((batch_size,), t), noise=noise)
            vb, _ = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=torch.full((batch_size,), t))
            total_vb += vb

        prior_b = self.prior_bpd(x_start=x_start)
        total_b = total_vb + prior_b

        return {
            'total': total_b,
            'vbterms': total_vb,
            'prior': prior_b
        }


encoder_model = Edge_Encoder
#encoder_model = Edge_Encoder_MLP
encoder_model = Edge_Encoder_Residual

nodes = [(0, {'pos': (0.1, 0.7)}),
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
         (26, {'pos': (0.95, 0.05)}),
         (27, {'pos': (0.75, 1.0)}),]

edges = [(0, 21), (0, 1), (0, 15), (21, 22), (22, 20), (20, 23), (23, 24), (24, 18), (19, 14), (14, 15), (15, 16), (16, 20), (19, 20), (19, 17), (14, 17), (14, 16), (17, 18), (12, 18), (12, 13), (13, 14), (10, 14), (1, 15), (9, 15), (1, 9), (1, 2), (11, 12), (9, 10), (3, 7), (2, 3), (7, 8), (8, 9), (8, 10), (10, 11), (8, 11), (6, 11), (3, 4), (4, 5), (4, 6), (5, 6), (24, 25), (12, 25), (5, 25), (11, 25), (5, 26), (23, 27), (24, 27)]


    
data_config = {"dataset": "tdrive_1000",
    "train_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_small.h5',
    "test_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_small.h5',
    "history_len": 5,
    "future_len": 2,
    "num_classes": 2,
    "edge_features": ['one_hot_edges'],
    "one_hot_nodes": False}

diffusion_config = {"type": 'linear', # Options: 'linear', 'cosine', 'jsd'
    "start": 0.9,  # 1e-4 gauss, 0.02 uniform
    "stop": 1.0,  # 0.02 gauss, 1. uniform
    "num_timesteps": 1000}

model_config = {"name": "edge_encoder_residual",
    "hidden_channels": 32,
    "time_embedding_dim": 16,
    "condition_dim": 16,
    "out_ch": 1,
    "num_heads": 2,
    "num_layers": 2,
    "dropout": 0.1,
    "model_output": "logits",
    "model_prediction": "x_start",  # Options: 'x_start','xprev'
    "transition_mat_type": 'gaussian',  # Options: 'gaussian','uniform','absorbing', 'marginal_prior'
    "transition_bands": 1,
    "loss_type": "cross_entropy_x_start",  # Options: kl, cross_entropy_x_start, hybrid
    "hybrid_coeff": 0.001,  # Only used for hybrid loss type.
    "class_weights": [0.1, 0.9] # = future_len/num_edges and (num_edges - future_len)/num_edges
    }

train_config = {"batch_size": 1,
    "optimizer": "adam",
    "lr": 0.01,
    "gradient_accumulation": True,
    "gradient_accumulation_steps": 4,
    "num_epochs": 10000,
    "learning_rate_warmup_steps": 2000, # previously 10000
    "lr_decay": 0.9999, # previously 0.9999
    "log_loss_every_steps": 10,
    "save_model": False,
    "save_model_every_steps": 1000}

test_config = {"batch_size": 1,
    "model_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/synthetic_d3pm_residual_fixed/synthetic_d3pm_residual_fixed_hidden_dim_32_time_dim_16_condition_dim_16_layers_2_weights_0.1.pth',
    "number_samples": 1
  }

wandb_config = {"exp_name": "synthetic_d3pm_test",
    "project": "trajectory_prediction_using_denoising_diffusion_models",
    "entity": "joeschmit99",
    "job_type": "test",
    "notes": "",
    "tags": ["synthetic", "edge_encoder"]} 

model = Graph_Diffusion_Model(data_config, diffusion_config, model_config, train_config, test_config, wandb_config, encoder_model, nodes, edges)
model.train()
model_path = test_config["model_path"]
sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=True, model_path=model_path, task='predict', number_samples=20)
print(sample_list)
print("\n")
print(ground_truth_hist)
print("\n")
print(ground_truth_fut)
model.visualize_predictions(sample_list, ground_truth_hist, ground_truth_fut)
