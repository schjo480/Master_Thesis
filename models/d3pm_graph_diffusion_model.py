import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
#from dataset.trajctory_dataset import TrajectoryDataset, collate_fn
from dataset.trajectory_dataset_geometric import TrajectoryGeoDataset, custom_collate_fn
from .d3pm_diffusion import make_diffusion
from tqdm import tqdm
import logging
import os
import time
import wandb
import networkx as nx
from torch_geometric.data import Data
from torch.profiler import profile, record_function, ProfilerActivity


class Graph_Diffusion_Model(nn.Module):
    def __init__(self, data_config, diffusion_config, model_config, train_config, test_config, wandb_config, model):
        super(Graph_Diffusion_Model, self).__init__()
        
        # Data
        self.data_config = data_config
        self.train_data_path = self.data_config['train_data_path']
        self.val_data_path = self.data_config['val_data_path']
        self.history_len = self.data_config['history_len']
        self.future_len = self.data_config['future_len']
        self.num_classes = self.data_config['num_classes']
        self.edge_features = self.data_config['edge_features']
        self.pos_encoding_dim = self.data_config['pos_encoding_dim']
        
        # Diffusion
        self.diffusion_config = diffusion_config
        self.num_timesteps = self.diffusion_config['num_timesteps']
        
        # Model
        self.model_config = model_config
        self.model = model # Edge_Encoder
        self.hidden_channels = self.model_config['hidden_channels']
        self.time_embedding_dim = self.model_config['time_embedding_dim']
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
        wandb.run.name = self.wandb_config['run_name']

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
        self.log_metrics_every_steps = self.train_config['log_metrics_every_steps']   
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build Components
        self._build_train_dataloader()
        self._build_val_dataloader()
        self._build_model()
        self._build_optimizer()
        
        # Move model to GPU
        
        self.model.to(self.device, non_blocking=True)
        print("device", self.device)
        
    def train(self):
        """
        Trains the diffusion-based trajectory prediction model.

        This function performs the training of the diffusion-based trajectory prediction model. It iterates over the specified number of epochs and updates the model's parameters based on the training data. The training process includes forward propagation, loss calculation, gradient computation, and parameter updates.

        Returns:
            None
        """
        torch.autograd.set_detect_anomaly(True)
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges, future_len=self.future_len, device=self.device)
        def model_fn(x, edge_index, t, condition=None):
            if self.model_config['name'] == 'edge_encoder':
                return self.model.forward(x, edge_index, t=t, condition=condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_residual':
                return self.model.forward(x, edge_index, t=t, condition=condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(x, t=t, condition=condition, mode='future')
                
        for epoch in tqdm(range(self.num_epochs)):
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"Epoch": epoch, "Learning Rate": current_lr})
            
            total_loss = 0
            ground_truth_fut = []
            pred_fut = []
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                #with record_function("model_training"):
            if self.gradient_accumulation:
                for data in self.train_data_loader:
                    history_edge_features = data["history_edge_features"]
                    history_indices = data["history_indices"]
                    future_edge_indices_one_hot = data["future_edge_features"][:, :, 0]
                    
                    self.optimizer.zero_grad()
                    for i in range(min(self.gradient_accumulation_steps, history_edge_features.size(0))):
                        # Calculate history condition c
                        if self.model_config['name'] == 'edge_encoder':
                            c = self.model.forward(x=history_edge_features[i].unsqueeze(0), edge_index=self.edge_index, indices=history_indices[i], mode='history')
                        elif self.model_config['name'] == 'edge_encoder_residual':
                            c = self.model.forward(x=history_edge_features[i].unsqueeze(0), edge_index=self.edge_index, indices=history_indices[i], mode='history')
                        elif self.model_config['name'] == 'edge_encoder_mlp':
                            c = self.model.forward(x=history_edge_features[i].unsqueeze(0), indices=history_indices[i], mode='history')
                        else:
                            raise NotImplementedError(self.model_config['name'])
                        
                        x_start = future_edge_indices_one_hot[i].unsqueeze(0)   # (1, num_edges)
                        # Get loss and predictions
                        loss, preds = dif.training_losses(model_fn, c, x_start=x_start, edge_features=history_edge_features[i].unsqueeze(0), edge_index=self.edge_index, line_graph=None)   # preds are of shape (num_edges,)
                        
                        total_loss += loss / self.gradient_accumulation_steps
                        (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                        
                        if epoch % 10 == 0:
                            ground_truth_fut.append(x_start.detach().to('cpu'))
                            pred_fut.append(preds.detach().to('cpu'))
                        
                    
                    self.optimizer.step()
                    
            else:
                for data in self.train_data_loader:
                    history_edge_features = data.x
                    history_indices = data.history_indices
                    future_edge_indices_one_hot = data.y[:, :, 0]
                    
                    batch_size = future_edge_indices_one_hot.size(0)
                    '''if self.model_config['name'] == 'edge_encoder_mlp':
                        if batch_size == self.batch_size:
                            future_edge_indices_one_hot = future_edge_indices_one_hot.view(self.batch_size, self.num_edges)
                        else:
                            future_edge_indices_one_hot = future_edge_indices_one_hot.view(batch_size, self.num_edges)'''
                    
                    self.optimizer.zero_grad()
                    # Calculate history condition c
                    if self.model_config['name'] == 'edge_encoder':
                        c = self.model.forward(x=history_edge_features, edge_index=data.edge_index, indices=history_indices, mode='history')
                    elif self.model_config['name'] == 'edge_encoder_residual':
                        c = self.model.forward(x=history_edge_features, edge_index=data.edge_index, indices=history_indices, mode='history')
                    elif self.model_config['name'] == 'edge_encoder_mlp':
                        c = self.model.forward(x=history_edge_features, indices=history_indices, mode='history')
                    else:
                        raise NotImplementedError(self.model_config['name'])
                    
                    x_start = future_edge_indices_one_hot   # (batch_size, num_edges, 1)
                    # Get loss and predictions
                    loss, preds = dif.training_losses(model_fn, c, x_start=x_start, edge_features=history_edge_features, edge_index=data.edge_index, line_graph=None)
                    total_loss += loss
                    loss.backward()
                    
                    self.optimizer.step()
                    if epoch % self.log_metrics_every_steps == 0:
                        ground_truth_fut.append(x_start.detach().to('cpu'))
                        pred_fut.append(preds.detach().to('cpu'))
            self.scheduler.step()
                    
            if epoch % self.log_loss_every_steps == 0:
                avg_loss = total_loss / len(self.train_data_loader)
                wandb.log({"Epoch": epoch, "Average Train Loss": avg_loss.item()})
                self.log.info(f"Epoch {epoch}, Average Train Loss: {avg_loss.item()}")
                print(f"Epoch {epoch}, Average Train Loss: {avg_loss.item()}")
                if epoch % self.log_metrics_every_steps == 0:
                    #print("Ground Truth Training", [[torch.argwhere(ground_truth_fut[i][j].unsqueeze(0) == 1)[:, 1] for i in range(len(ground_truth_fut))] for j in range(ground_truth_fut[0].size(0))])
                    #print("Samples Training", [[torch.argwhere(pred_fut[i][j].unsqueeze(0) == 1)[:, 1] for i in range(len(pred_fut))] for j in range(pred_fut[0].size(0))])
                    f1_score = F1Score(task='binary', average='macro', num_classes=2)
                    f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach().to('cpu'), torch.flatten(torch.cat(ground_truth_fut)).detach().to('cpu'))
                    print("Train F1 Score:", f1_epoch.item())
                    wandb.log({"Epoch": epoch, "Train F1 Score": f1_epoch.item()})
                    #print("Samples", pred_fut)
                    #print("Sample 1", pred_fut[0])
                    #print("Sample 1", pred_fut[0].size())
                    #print([sum(sum(sample) for sample in pred_fut[i] / len(pred_fut[i])) for i in range(len(pred_fut))])
                    #intermediate = [sum(sum(sample) for sample in pred_fut[i]) / len(pred_fut[i]) for i in range(len(pred_fut))]
                    #avg_sample_length_train = sum(intermediate) / len(intermediate)
                    #wandb.log({"Epoch": epoch, "Average train sample length": avg_sample_length_train.item()})
                
            if (epoch + 1) % self.eval_every_steps == 0:
                print("Evaluating on validation set...")
                sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = self.get_samples(task='predict')
                fut_ratio, f1, avg_sample_length = self.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary)
                print("Samples", sample_list)
                print("Ground truth", ground_truth_fut)
                print("Val F1 Score", f1.item())
                print("Average val sample length", round(avg_sample_length, 3))
                wandb.log({"Epoch": epoch, "Val F1 Score": f1.item()})
                wandb.log({"Epoch": epoch, "Val Future ratio": fut_ratio})
                wandb.log({"Epoch": epoch, "Average val sample length": round(avg_sample_length, 3)})
                        
            if self.train_config['save_model'] and (epoch + 1) % self.train_config['save_model_every_steps'] == 0:
                self.save_model()
            
    def get_samples(self, load_model=False, model_path=None, task='predict', number_samples=1, save=False):
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
        
        def model_fn(x, edge_index, t, condition=None):
            if self.model_config['name'] == 'edge_encoder':
                return self.model.forward(x, edge_index, t=t, condition=condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_residual':
                return self.model.forward(x, edge_index, t=t, condition=condition, mode='future')
            elif self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(x=x, t=t, condition=condition, mode='future')
        
        sample_binary_list = []
        sample_list = []
        ground_truth_hist = []
        ground_truth_fut = []
        ground_truth_fut_binary = []
        
        if task == 'predict':
            for data in tqdm(self.val_dataloader):
                history_edge_features = data.x

                history_edge_indices = data.history_indices

                future_trajectory_indices = data.future_indices
                future_edge_indices_one_hot = data.y[:, :, 0]
                
                # with torch.no_grad():
                if self.model_config['name'] == 'edge_encoder':
                    c = self.model.forward(x=history_edge_features, edge_index=data.edge_index, indices=history_edge_indices, mode='history')
                elif self.model_config['name'] == 'edge_encoder_residual':
                    c = self.model.forward(x=history_edge_features, edge_index=data.edge_index, indices=history_edge_indices, mode='history')
                elif self.model_config['name'] == 'edge_encoder_mlp':
                    c = self.model.forward(x=history_edge_features, indices=history_edge_indices, mode='history')
            
                if number_samples > 1:
                    new_seed = torch.seed() + torch.randint(0, 100000, (1,)).item()
                    torch.manual_seed(new_seed)
                    sample_sublist = []
                    for _ in range(number_samples):
                        samples = make_diffusion(self.diffusion_config, self.model_config, 
                                                num_edges=self.num_edges, future_len=self.future_len).p_sample_loop(model_fn=model_fn,
                                                                                        shape=(self.test_batch_size, self.num_edges),
                                                                                        edge_features=history_edge_features,
                                                                                        edge_index=self.edge_index,
                                                                                        line_graph=None,
                                                                                        condition=c,
                                                                                        task=task)
                        samples = torch.where(samples == 1)[1]
                        sample_sublist.append(samples.detach().to('cpu'))
                    sample_list.append(sample_sublist)

                elif number_samples == 1:
                    samples_binary = make_diffusion(self.diffusion_config, self.model_config, 
                                            num_edges=self.num_edges, future_len=self.future_len, device=self.device).p_sample_loop(model_fn=model_fn,
                                                                                    shape=(self.test_batch_size, self.num_edges), 
                                                                                    edge_features=history_edge_features,
                                                                                    edge_index=data.edge_index,
                                                                                    line_graph=None,
                                                                                    condition=c,
                                                                                    task=task)
                    sample_binary_list.append(samples_binary)
                    samples = torch.argwhere(samples_binary == 1)[:, 1]
                    sample_list.append(samples.detach().to('cpu'))
                else:
                    raise ValueError("Number of samples must be greater than 0.")
                ground_truth_hist.append(history_edge_indices.detach().to('cpu'))
                ground_truth_fut.append(future_trajectory_indices.detach().to('cpu'))
                ground_truth_fut_binary.append(future_edge_indices_one_hot.detach().to('cpu'))
            
            if number_samples == 1:
                pass
                #fut_ratio, f1, avg_sample_length = self.eval(sample_list, ground_truth_hist, ground_truth_fut)
                #wandb.log({"F1 Score": f1.item()})
                #wandb.log({"Future ratio": fut_ratio})
                #wandb.log({"Average sample length": avg_sample_length})
            
            if save:
                save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + '_' +  self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                 f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}_condition_dim_{self.condition_dim}_layers_{self.num_layers}')
                torch.save(sample_list, os.path.join(save_path, f'{self.exp_name}_samples.pth'))
                torch.save(ground_truth_hist, os.path.join(save_path, f'{self.exp_name}_ground_truth_hist.pth'))
                torch.save(ground_truth_fut, os.path.join(save_path, f'{self.exp_name}_ground_truth_fut.pth'))
                print(f"Samples saved at {os.path.join(save_path, f'{self.exp_name}_samples.pth')}!")
            else:
                return sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary
        
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

        sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = self.get_samples(load_model=True, model_path=self.test_config['model_path'], number_samples=number_samples)
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
    
    def eval(self, sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary):
        """
        Evaluate the model's performance.

        :param sample_list: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        """
        def calculate_fut_ratio(sample_list, ground_truth_fut):
            """
            Calculates the ratio of samples in `sample_list` that have at least n edges in common with the ground truth future trajectory for each n up to future_len.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                dict: A dictionary where keys are the minimum number of common edges (from 1 to future_len) and values are the ratios of samples meeting that criterion.
            """
            # Initialize counts for each number of common edges from 1 up to future_len
            counts = {i: 0 for i in range(1, self.future_len + 1)}
            total = len(sample_list)

            for i, sample in enumerate(sample_list):
                # Convert tensors to lists if they are indeed tensors
                sample = sample.tolist() if isinstance(sample, torch.Tensor) else sample
                ground_truth = ground_truth_fut[i].flatten().tolist()

                edges_count = sum(1 for edge in ground_truth if edge in sample)
                for n in range(1, min(edges_count, self.future_len) + 1):
                    counts[n] += 1

            ratios = {n: counts[n] / total for n in counts}
            return ratios
        
        def calculate_sample_f1(sample_binary_list, ground_truth_fut_binary):
            """
            Calculates the F1 score for a given list of samples and ground truth futures.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth futures.

            Returns:
                float: The F1 score.

            """
            '''one_hot_samples = [torch.zeros(self.num_edges) for _ in range(len(sample_list))]
            one_hot_futures = [torch.zeros(self.num_edges) for _ in range(len(ground_truth_fut))]
            for i, one_hot_sample in enumerate(one_hot_samples):
                for edge_index, edge in enumerate(self.edges):
                    if edge_index in sample_list[i]:
                        one_hot_sample[edge_index] = 1
            for i, one_hot_fut in enumerate(one_hot_futures):
                for edge_index, edge in enumerate(self.edges):
                    if edge_index in ground_truth_fut[i]:
                        one_hot_fut[edge_index] = 1'''
            metric = F1Score(task='binary', average='macro', num_classes=2)
            f1 = metric(torch.flatten(torch.cat(sample_binary_list)).detach().to('cpu'),
                        torch.flatten(torch.cat(ground_truth_fut_binary)).detach().to('cpu'))
            

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
        f1 = calculate_sample_f1(sample_binary_list, ground_truth_fut_binary)
        avg_sample_length = calculate_avg_sample_length(sample_list)
        
        return fut_ratio, f1, avg_sample_length
    
    def save_model(self):
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + '_' + f'_hist{self.history_len}' + f'_fut{self.future_len}_' + self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                 f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}_condition_dim_{self.condition_dim}.pth')
        torch.save(self.model.state_dict(), save_path)
        self.log.info(f"Model saved at {save_path}!")
        print(f"Model saved at {save_path}")
        
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
        # self.train_dataset = TrajectoryDataset(self.train_data_path, self.history_len, self.future_len, self.edge_features, device=self.device)
        self.train_dataset = TrajectoryGeoDataset(self.train_data_path, self.history_len, self.future_len, self.edge_features, device=self.device)
        self.G = self.train_dataset.build_graph()
        self.nodes = self.G.nodes
        self.edges = self.G.edges(data=True)
        self.num_edges = self.G.number_of_edges()
        self.indexed_edges = self.train_dataset.edges
        # self.num_edge_features = self.train_dataset.num_edge_features
        self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features += 4
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features += 1
        
        # Build the line graph and corresponding edge index
        # self.edge_index = self._build_edge_index()
        # self.edge_index = self.train_dataset.edge_index
        
        from torch_geometric.data import DataLoader
        self.train_data_loader = DataLoader(self.train_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=False, 
                                            collate_fn=custom_collate_fn, 
                                            num_workers=0,
                                            pin_memory=False,
                                            follow_batch=['x', 'y', 'history_indices', 'future_indices'])
                        
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
        
        return edge_index.to(self.device, non_blocking=True)
    
    def _build_val_dataloader(self):
        # self.val_dataset = TrajectoryDataset(self.val_data_path, self.history_len, self.future_len, self.edge_features, device=self.device)
        self.val_dataset = TrajectoryGeoDataset(self.val_data_path, self.history_len, self.future_len, self.edge_features, device=self.device)
        from torch_geometric.data import DataLoader
        self.val_dataloader = DataLoader(self.val_dataset, 
                                         batch_size=self.test_batch_size, 
                                         shuffle=False, 
                                         collate_fn=custom_collate_fn,
                                         follow_batch=['x', 'y', 'history_indices', 'future_indices']
                                         )
        print("> Test Dataset loaded!")
        
    def _build_model(self):
        self.model = self.model(self.model_config, self.history_len, self.future_len, self.num_classes,
                                num_edges=self.num_edges, hidden_channels=self.hidden_channels, edge_features=self.edge_features, num_edge_features=self.num_edge_features, num_timesteps=self.num_timesteps, pos_encoding_dim=self.pos_encoding_dim)
        print("> Model built!")
             