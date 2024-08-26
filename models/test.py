import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
#from dataset.trajectory_dataset_geometric import TrajectoryGeoDataset, custom_collate_fn
#from .d3pm_diffusion import make_diffusion
from tqdm import tqdm
import logging
import os
import time
import wandb
import networkx as nx
torch.set_printoptions(threshold=10_000)

class Graph_Diffusion_Model(nn.Module):
    def __init__(self, data_config, diffusion_config, model_config, train_config, test_config, wandb_config, model, pretrained=False):
        super(Graph_Diffusion_Model, self).__init__()
        
        # Data
        self.data_config = data_config
        self.train_data_path = self.data_config['train_data_path']
        self.val_data_path = self.data_config['val_data_path']
        self.history_len = self.data_config['history_len']
        self.future_len = self.data_config['future_len']
        self.num_classes = self.data_config['num_classes']
        self.edge_features = self.data_config['edge_features']
        
        # Diffusion
        self.diffusion_config = diffusion_config
        self.num_timesteps = self.diffusion_config['num_timesteps']
        
        # Model
        self.model_config = model_config
        self.model = model # Edge_Encoder
        self.hidden_channels = self.model_config['hidden_channels']
        self.time_embedding_dim = self.model_config['time_embedding_dim']
        self.num_layers = self.model_config['num_layers']
        self.pos_encoding_dim = self.model_config['pos_encoding_dim']
        
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
        features = ''
        for feature in self.edge_features:
            features += feature + '_'
        run_name = f'{self.exp_name}_{self.model_config['transition_mat_type']}_{self.diffusion_config['type']}_{features}_hist{self.history_len}_fut{self.future_len}'
        wandb.run.name = run_name

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
        
        # Load model
        if pretrained:
            features = ''
            for feature in self.edge_features:
                features += feature + '_'
            pretrained_model_path = os.path.join(self.model_dir, 
                                    self.exp_name + '_' + self.model_config['name'] + features + '_' + f'_hist{self.history_len}' + f'_fut{self.future_len}_' + self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                    f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}.pth')
            self.load_model(pretrained_model_path)
        
        # Move model to GPU
        
        self.model.to(self.device, non_blocking=True)
        print("Device", self.device)
        
    def train(self):
        """
        Trains the diffusion-based trajectory prediction model.

        This function performs the training of the diffusion-based trajectory prediction model. It iterates over the specified number of epochs and updates the model's parameters based on the training data. The training process includes forward propagation, loss calculation, gradient computation, and parameter updates.

        Returns:
            None
        """
        torch.autograd.set_detect_anomaly(True)
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges, future_len=self.future_len, edge_features=self.edge_features, device=self.device)
        def model_fn(x, edge_index, t, indices=None):
            if self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(x, t=t, indices=indices)
            else:
                return self.model.forward(x, edge_index, t=t, indices=indices)
                
        for epoch in tqdm(range(self.num_epochs)):
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"Epoch": epoch, "Learning Rate": current_lr})
            
            total_loss = 0
            acc = 0
            tpr = 0
            prec = 0
            ground_truth_fut = []
            pred_fut = []

            if self.gradient_accumulation:
                for data in self.train_data_loader:
                    edge_features = data["edge_features"]
                    history_indices = data["history_indices"]
                    future_edge_indices_one_hot = data["future_edge_features"][:, :, 0]
                    
                    self.optimizer.zero_grad()
                    for i in range(min(self.gradient_accumulation_steps, edge_features.size(0))):
                        # Calculate history condition c
                        if self.model_config['name'] == 'edge_encoder':
                            c = self.model.forward(x=edge_features[i].unsqueeze(0), edge_index=self.edge_index, indices=history_indices[i])
                        elif self.model_config['name'] == 'edge_encoder_residual':
                            c = self.model.forward(x=edge_features[i].unsqueeze(0), edge_index=self.edge_index, indices=history_indices[i])
                        elif self.model_config['name'] == 'edge_encoder_mlp':
                            c = self.model.forward(x=edge_features[i].unsqueeze(0), indices=history_indices[i])
                        else:
                            raise NotImplementedError(self.model_config['name'])
                        
                        x_start = future_edge_indices_one_hot[i].unsqueeze(0)   # (1, num_edges)
                        # Get loss and predictions
                        loss, preds = dif.training_losses(model_fn, c, x_start=x_start, edge_features=edge_features[i].unsqueeze(0), edge_index=self.edge_index, line_graph=None)   # preds are of shape (num_edges,)
                        
                        total_loss += loss / self.gradient_accumulation_steps
                        (loss / self.gradient_accumulation_steps).backward() # Gradient accumulation
                        
                        if epoch % 10 == 0:
                            ground_truth_fut.append(x_start.detach().to('cpu'))
                            pred_fut.append(preds.detach().to('cpu'))
                        
                    self.optimizer.step()
                    
            else:
                for data in self.train_data_loader:
                    edge_features = data.x      # (batch_size * num_edges, num_edge_features)
                    history_indices = data.history_indices  # (batch_size, history_len)
                    x_start = data.y[:, :, 0]   # (batch_size, num_edges)
                    edge_index = data.edge_index    # (2, batch_size * num_edges)
                                        
                    self.optimizer.zero_grad()
                    
                    # Get loss and predictions
                    loss, preds = dif.training_losses(model_fn, x_start=x_start, edge_features=edge_features, edge_index=edge_index, indices=history_indices)
                    total_loss += loss
                    # Gradient calculation and optimization
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.3)
                    self.optimizer.step()
                    
                    if epoch % self.log_metrics_every_steps == 0:
                        acc += torch.sum(preds == x_start).item() / (x_start.size(0) * x_start.size(1))
                        tpr += torch.sum(preds * x_start).item() / torch.sum(x_start).item()
                        if torch.sum(preds) > 0:
                            prec += torch.sum(preds * x_start).item() / torch.sum(preds).item()
                        ground_truth_fut.append(x_start.detach().to('cpu'))
                        pred_fut.append(preds.detach().to('cpu'))
            self.scheduler.step()
            print("Number of invalid samples:", self.train_dataset.ct // (epoch + 1))
            
            # Log Loss
            if epoch % self.log_loss_every_steps == 0:
                avg_loss = total_loss / len(self.train_data_loader)
                print(f"Epoch {epoch}, Average Train Loss: {avg_loss.item()}")
                wandb.log({"Epoch": epoch, "Average Train Loss": avg_loss.item()})
            
            # Log Metrics
            if epoch % self.log_metrics_every_steps == 0:
                avg_acc = acc / len(self.train_data_loader)
                avg_tpr = tpr / len(self.train_data_loader)
                avg_prec = prec / len(self.train_data_loader)
                if (avg_prec + avg_tpr) > 0:
                    f1_score = 2 * (avg_prec * avg_tpr) / (avg_prec + avg_tpr)
                else:
                    f1_score = 0
                print("Train F1 Score:", f1_score)
                print("Train Accuracy:", avg_acc)
                print("Train TPR:", avg_tpr)
                wandb.log({"Epoch": epoch, "Train F1 Score": f1_score})
                wandb.log({"Epoch": epoch, "Train Accuracy": avg_acc})
                wandb.log({"Epoch": epoch, "Train TPR": avg_tpr})
            
            # Validation
            if (epoch + 1) % self.eval_every_steps == 0:
                print("Evaluating on validation set...")
                sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = self.get_samples(task='predict', number_samples=1)
                fut_ratio, f1, val_acc, val_tpr, avg_sample_length, valid_sample_ratio = self.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=1)
                #print("Samples:", sample_list)
                #print("Ground truth:", ground_truth_fut)
                print("Val F1 Score:", f1)
                print("Val Accuracy:", round(val_acc, 6))
                print("Val TPR:", round(val_tpr, 6))
                print("Average val sample length:", round(avg_sample_length, 3))
                print("Valid sample ratio:", round(valid_sample_ratio, 3))
                wandb.log({"Epoch": epoch, "Val F1 Score": f1})
                wandb.log({"Epoch": epoch, "Val Accuracy": val_acc})
                wandb.log({"Epoch": epoch, "Val TPR": val_tpr})
                wandb.log({"Epoch": epoch, "Val Future ratio": fut_ratio})
                wandb.log({"Epoch": epoch, "Average val sample length": round(avg_sample_length, 3)})
                wandb.log({"Epoch": epoch, "Valid sample ratio": round(valid_sample_ratio, 3)})
                        
            if self.train_config['save_model'] and (epoch + 1) % self.train_config['save_model_every_steps'] == 0:
                self.save_model()
            
    def get_samples(self, load_model=False, model_path=None, task='predict', save=False, number_samples=None):
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
        
        if number_samples is None:
            number_samples = self.test_config['number_samples']
        
        def model_fn(x, edge_index, t, indices=None):
            if self.model_config['name'] == 'edge_encoder_mlp':
                return self.model.forward(x, t=t, indices=indices)
            else:
                return self.model.forward(x, edge_index, t=t, indices=indices)
        
        sample_binary_list = []
        sample_list = []
        ground_truth_hist = []
        ground_truth_fut = []
        ground_truth_fut_binary = []
        
        if task == 'predict':
            for data in tqdm(self.val_dataloader):
                edge_features = data.x
                bs = edge_features.size(0) // self.num_edges
                history_edge_indices = data.history_indices
                future_trajectory_indices = data.future_indices
                future_binary = data.y[:, :, 0]
                edge_index = data.edge_index
            
                if number_samples > 1:
                    sample_sublist_binary = []
                    sample_sublist = []
                    new_seed = torch.seed() + torch.randint(0, 100000, (1,)).item()
                    torch.manual_seed(new_seed)
                    for _ in range(number_samples):
                        samples = make_diffusion(self.diffusion_config, 
                                                 self.model_config, 
                                                 num_edges=self.num_edges, 
                                                 future_len=self.future_len,
                                                 edge_features=self.edge_features,
                                                 device=self.device).p_sample_loop(model_fn=model_fn,
                                                                                   shape=(bs, self.num_edges),
                                                                                   edge_features=edge_features,
                                                                                   edge_index=edge_index,
                                                                                   indices=history_edge_indices,
                                                                                   task=task)
                        sample_sublist_binary.append(samples.detach().to('cpu'))
                        if self.test_batch_size == 1:
                            samples_list = torch.argwhere(samples == 1)[:, 1].detach().to('cpu')
                        else:
                            samples_list = []
                            for i in range(samples.size(0)):
                                sample = torch.argwhere(samples[i] == 1).flatten().to('cpu')
                                samples_list.append(sample)
                        
                        sample_sublist.append(samples_list)
                    sample_list.append(sample_sublist)
                    sample_binary_list.append(sample_sublist_binary)
                
                elif number_samples == 1:
                    samples_binary = make_diffusion(self.diffusion_config, 
                                                    self.model_config, 
                                                    num_edges=self.num_edges, 
                                                    future_len=self.future_len,
                                                    edge_features=self.edge_features,
                                                    device=self.device).p_sample_loop(model_fn=model_fn,
                                                                                    shape=(bs, self.num_edges), 
                                                                                    edge_features=edge_features,
                                                                                    edge_index=edge_index,
                                                                                    indices=history_edge_indices,
                                                                                    task=task)
                    sample_binary_list.append(samples_binary.to('cpu'))
                    
                    # Updated logic to handle sample extraction
                    current_samples = []  # This will store tensors for the current batch
                    for batch_index in range(samples_binary.shape[0]):
                        indices = (samples_binary[batch_index] == 1).nonzero(as_tuple=False).squeeze(-1)
                        # Append the indices directly; no need to check for emptiness as it creates an empty tensor automatically if no '1's are found
                        current_samples.append(indices.detach().to('cpu'))
                    
                    # Append the list of tensors for this batch to the main sample list
                    sample_list.append(current_samples)
                else:
                    raise ValueError("Number of samples must be greater than 0.")
                
                ground_truth_hist.append(history_edge_indices.detach().to('cpu'))
                ground_truth_fut.append(future_trajectory_indices.detach().to('cpu'))
                ground_truth_fut_binary.append(future_binary.detach().to('cpu'))
            
            if save:
                save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + '_' +  self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                 f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}_layers_{self.num_layers}')
                torch.save(sample_list, os.path.join(save_path, f'{self.exp_name}_samples.pth'))
                torch.save(ground_truth_hist, os.path.join(save_path, f'{self.exp_name}_ground_truth_hist.pth'))
                torch.save(ground_truth_fut, os.path.join(save_path, f'{self.exp_name}_ground_truth_fut.pth'))
                print(f"Samples saved at {os.path.join(save_path, f'{self.exp_name}_samples.pth')}!")
            else:
                ### TODO: Debugging ###
                # ground_truth_fut is a list (of length len(val_dataloader)) of tensors with shape (batch_size, future_len)
                # ground_truth_fut_binary is a list (of length len(val_dataloader)) of tensors with shape (batch_size, num_edges)
                
                # For num_samples = 1 and batch_size > 1: (works ok!)
                # sample_binary_list is a list (of length len(val_dataloader)) of tensors with shape (batch_size, num_edges)
                # (sample_list is a list (of length len(val_dataloader)) of lists (of length batch_size) of tensors with size (num_edges == 1) (i.e could be empty tensors))
                
                # For num_samples > 1 and batch_size > 1: (Problem with calculating statistics after training)
                # sample_binary_list is a list (of length len(val_dataloader)) of lists (of length number_samples) of tensors with shape (batch_size, num_edges)
                # sample_list is a list (of length len(val_dataloader) of lists (of length number_samples) of lists (of length batch_size) of tensors with size (num_edges == 1) (i.e could be empty tensors)
                
                # For num_samples = 1 and batch_size = 1: (Problem with calculating statistics during training)
                # sample_binary_list is a list (of length len(val_dataloader), i.e. number of trajectories here) of tensors with shape (batch_size=1, num_edges)
                # sample_list is a list (of length len(val_dataloader), i.e. number of trajectories here) of tensors with size (num_edges == 1) (i.e could be empty tensors), no batch dimension as in ground_truth_fut
                
                # For num_samples > 1 and batch_size = 1: (Problem with calculating statistics after training)
                # sample_binary_list is a list (of length len(val_dataloader), i.e. number of trajectories here) of lists (of length number_samples) of tensors with shape (batch_size=1, num_edges)
                # sample_list is a list (of length len(val_dataloader), i.e. number of trajectories here) of lists (of length number_samples) of tensors with size (num_edges == 1) (i.e could be empty tensors), no batch dimension as in ground_truth_fut
                
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
    
    def eval(self, sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=None, return_samples=False):
        """
        Evaluate the model's performance.

        :param sample_list: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        """
        def check_sample(sample):
                """
                Check if the sample is valid, i.e. connected, acyclical, and no splits.

                :param sample: A list of predicted edge indices.
                """
                if len(sample) == 0:
                    return True, sample
                
                else:
                    # Check connectivity
                    def is_connected_sequence(edge_indices, graph):
                        if len(edge_indices) <= 1:
                            return True
                        edges = self.train_dataset.indexed_edges
                        subgraph_nodes = set()
                        for idx in edge_indices:
                            edge = edges[idx][0]  # get the node tuple for each edge
                            subgraph_nodes.update(edge)

                        # Create a subgraph with these nodes
                        subgraph = graph.subgraph(subgraph_nodes)

                        # Check if the subgraph is connected
                        return nx.is_connected(subgraph)

                    # Check acyclical
                    def is_acyclical_sequence(edge_indices, graph):
                        if len(edge_indices) <= 1:
                            return True
                        edges = self.train_dataset.indexed_edges
                        subgraph_nodes = []
                        subgraph_edges = []
                        for idx in edge_indices:
                            edge = edges[idx][0]  # get the node tuple for each edge
                            subgraph_nodes.append(edge)
                            subgraph_edges.append(edge)

                        subgraph = nx.Graph()
                        subgraph.add_edges_from(subgraph_edges)
                        
                        has_cycle = nx.cycle_basis(subgraph)
                        return len(has_cycle) == 0
                    
                    # Check no splits
                    def is_not_split(edge_indices, graph):
                        if len(edge_indices) <= 1:
                            return True
                        edges = self.train_dataset.indexed_edges
                        subgraph_nodes = set()
                        subgraph_edges = []
                        for idx in edge_indices:
                            edge = edges[idx][0]  # get the node tuple for each edge
                            subgraph_nodes.update(edge)
                            subgraph_edges.append(edge)

                        # Create a directed version of the subgraph to check for cycles
                        graph = nx.Graph()
                        graph.add_nodes_from(subgraph_nodes)
                        graph.add_edges_from(subgraph_edges)
                        
                        if any(graph.degree(node) > 2 for node in graph.nodes()):
                            return False
                        else:
                            return True

                    connected = is_connected_sequence(sample, self.G)
                    acyclical = is_acyclical_sequence(sample, self.G)
                    no_splits = is_not_split(sample, self.G)
                    
                    #print("Connected", connected)
                    #print("Acyclical", acyclical)
                    #print("No splits", no_splits)
                    if connected and acyclical and no_splits:
                        return True, sample
                    else:
                        return False, sample
                    
        if number_samples is None:
            number_samples = self.test_config['number_samples']
        valid_sample_ratio = 0
        if number_samples > 1:
            valid_samples = []
            binary_valid_samples = []
            valid_ids = []
            valid_ct = 0
            #for batch_idx in tqdm(range(len(sample_list))):
            for i in range(len(sample_list)):
                valid_sample_list = []
                binary_valid_sample_list = []
                valid_id_list = []
                
                transposed_data = list(zip(*sample_list[i]))
                for j in range(len(transposed_data)):
                    preds = transposed_data[j]
                    for i, sample in enumerate(preds):
                        binary_valid_sample = torch.zeros(self.num_edges, device=sample.device)
                        valid_index = None
                        valid_sample = torch.tensor([])
                        valid, sample = check_sample(sample)
                        if valid:
                            valid_sample = sample
                            binary_valid_sample[valid_sample] = 1
                            valid_index = i
                            valid_ct += 1 / len(transposed_data)
                            break
                    if valid_index is None:
                        random_index = torch.randint(0, len(preds), (1,)).item()
                        random_sample = preds[random_index]
                        binary_valid_sample[random_sample] = 1
                        valid_id_list.append(valid_index)
                        valid_sample_list.append(random_sample)
                    else:
                        valid_id_list.append(valid_index)
                        valid_sample_list.append(valid_sample)
                        binary_valid_sample_list.append(binary_valid_sample)
                
                binary_valid_sample_list = torch.stack(binary_valid_sample_list, dim=0)
                valid_ids.append(valid_id_list)
                valid_samples.append(valid_sample_list)
                binary_valid_samples.append(binary_valid_sample_list)
                
            valid_sample_ratio = valid_ct / len(sample_list)
            def get_most_predicted_numbers(sample_list):
                """
                Get the 'self.future_len' distinct numbers that are predicted the most amount of time from each list in the input list.

                Args:
                    sample_list (list): A list of lists containing predicted numbers.

                Returns:
                    list: A list of tensors containing the most predicted numbers.
                """
                if self.test_batch_size == 1:
                    most_predicted_numbers = []
                    most_predicted_binary = []
                    for sample in sample_list:
                        binary_sample = torch.zeros(self.num_edges)
                        # Flatten the sample list
                        flattened_sample = torch.cat(sample).flatten()
                        # Count the occurrences of each number
                        counts = torch.bincount(flattened_sample, minlength=self.num_edges)
                        valid_indices = counts > 0
                        non_zero_indices = torch.arange(self.num_edges)[valid_indices]
                        sorted_non_zero_indices = non_zero_indices[torch.argsort(counts[non_zero_indices], descending=True)]
                        most_predicted_indices = sorted_non_zero_indices[:min(self.future_len, len(sorted_non_zero_indices))]
                        
                        most_predicted_numbers.append([most_predicted_indices])
                        binary_sample[most_predicted_indices] = 1
                        most_predicted_binary.append(binary_sample.unsqueeze(0))
                else:
                    most_predicted_numbers = []
                    most_predicted_binary = []
                    for i in range(len(sample_list)):
                        # Flatten the sample list
                        transposed_data = list(zip(*sample_list[i]))
                        # Concatenate the tensors in each group
                        flattened_sample = [torch.cat(tensors) for tensors in transposed_data]
                        # Initialize the list to hold binary representations for each batch item
                        batch_binary_samples = []
                        # Count the occurrences of each number
                        most_predicted_numbers_int = []
                        for s in flattened_sample:
                            binary_sample = torch.zeros(self.num_edges, device=s.device)  # Ensure the binary_sample is on the correct device
                            counts = torch.bincount(s, minlength=self.num_edges)
                            valid_indices = counts > 0
                            non_zero_indices = torch.arange(self.num_edges)[valid_indices]
                            sorted_non_zero_indices = non_zero_indices[torch.argsort(counts[non_zero_indices], descending=True)]
                            most_predicted_indices = sorted_non_zero_indices[:min(self.future_len, len(sorted_non_zero_indices))]
                            most_predicted_numbers_int.append(most_predicted_indices)
                            binary_sample[most_predicted_indices] = 1
                            # Append the binary representation for this sample to the batch list
                            batch_binary_samples.append(binary_sample)
                        
                        # Stack all binary samples for the current batch to match the desired shape (batch_size, num_edges)
                        batch_binary_tensor = torch.stack(batch_binary_samples, dim=0)
                        most_predicted_numbers.append(most_predicted_numbers_int)
                        most_predicted_binary.append(batch_binary_tensor)

                return most_predicted_numbers, most_predicted_binary

            # sample_list, sample_binary_list = get_most_predicted_numbers(sample_list)
            sample_list = valid_samples
            sample_binary_list = binary_valid_samples
        
        elif number_samples == 1:
            valid_ct = 0
            for sample_sublist in sample_list:
                for sample in sample_sublist:
                    valid, sample = check_sample(sample)
                    if valid:
                        valid_ct += 1 / len(sample_sublist)
            valid_sample_ratio = valid_ct / len(self.val_dataloader)
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

            total = 0
            for (sample_sublist, ground_truth_sublist) in zip(sample_list, ground_truth_fut):
                total += len(sample_sublist)
                for i in range(len(sample_sublist)):
                    # Convert tensors to lists if they are indeed tensors
                    sample = sample_sublist[i].tolist()
                    ground_truth = ground_truth_sublist[i].flatten().tolist()

                    edges_count = sum(1 for edge in ground_truth if edge in sample)
                    for n in range(1, min(edges_count, self.future_len) + 1):
                        counts[n] += 1
                
            if total != 0:
                ratios = {n: counts[n] / total for n in counts}
            else:
                ratios = {n: 0 for n in counts}
            return ratios
        
        def calculate_sample_f1(tpr, prec):
            """
            Calculates the F1 score for a given list of samples and ground truth futures.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth futures.

            Returns:
                float: The F1 score.

            """
            if (prec + tpr) > 0:
                f1_score = 2 * (prec * tpr) / (prec + tpr)
            else:
                f1_score = 0
            return f1_score
        
        def calculate_sample_accuracy(sample_binary_list, ground_truth_fut_binary):
            """
            Calculate the accuracy of the samples.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                float: The accuracy of the samples.
            """
            acc = 0
            for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                ground_truth = ground_truth_sublist.flatten()
                sample = sample_sublist.flatten()
                acc += torch.sum(sample == ground_truth).item() / ground_truth.size(0)
            return acc / len(self.val_dataloader)
        
        def calculate_sample_tpr(sample_binary_list, ground_truth_fut_binary):
            """
            Calculate the true positive rate of the samples.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                float: The true positive rate of the samples.
            """
            tpr = 0
            for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                ground_truth = ground_truth_sublist.flatten()
                sample = sample_sublist.flatten()
                if torch.sum(ground_truth) > 0:
                    tpr += torch.sum(sample * ground_truth).item() / torch.sum(ground_truth).item()
            return tpr / len(self.val_dataloader)
        
        def calculate_sample_prec(sample_binary_list, ground_truth_fut_binary):
            prec = 0
            for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                ground_truth = ground_truth_sublist.flatten()
                sample = sample_sublist.flatten()
                if torch.sum(sample) > 0:
                    prec += torch.sum(sample * ground_truth).item() / torch.sum(sample).item()
            return prec / len(self.val_dataloader)
        
        def calculate_avg_sample_length(sample_list):
            """
            Calculate the average sample length.

            Args:
                sample_list (list): A list of samples.

            Returns:
                float: The average sample length.
            """
            total_len = 0
            for sample_sublist in sample_list:
                for sample in sample_sublist:
                    total_len += len(sample) / len(sample_sublist)
            return total_len / len(self.val_dataloader)
        
        fut_ratio = calculate_fut_ratio(sample_list, ground_truth_fut)
        tpr = calculate_sample_tpr(sample_binary_list, ground_truth_fut_binary)
        prec = calculate_sample_prec(sample_binary_list, ground_truth_fut_binary)
        f1 = calculate_sample_f1(tpr, prec)
        acc = calculate_sample_accuracy(sample_binary_list, ground_truth_fut_binary)
        avg_sample_length = calculate_avg_sample_length(sample_list)
        
        if return_samples:
            return fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, sample_list, valid_ids, ground_truth_hist, ground_truth_fut
        else:
            return fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio
    
    def save_model(self):
        features = ''
        for feature in self.edge_features:
            features += feature + '_'
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + features + '_' + f'_hist{self.history_len}' + f'_fut{self.future_len}_' + self.model_config['transition_mat_type'] + '_' +  self.diffusion_config['type'] + 
                                 f'_hidden_dim_{self.hidden_channels}_time_dim_{str(self.time_embedding_dim)}.pth')
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
                return max(decay_lr, 4e-4 / self.lr)
            
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
        self.num_edge_features = 2  # Binary History and noised binary future
        if 'coordinates' in self.edge_features:
            self.num_edge_features += 4
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features += 1
        if 'road_type' in self.edge_features:
            self.num_edge_features += self.train_dataset.num_road_types
        if 'pw_distance' in self.edge_features:
            self.num_edge_features += 1
        if 'edge_length' in self.edge_features:
            self.num_edge_features += 1
        if 'edge_angles' in self.edge_features:
            self.num_edge_features += 1
        if 'num_pred_edges' in self.edge_features:
            self.num_edge_features += 1
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
        self.val_dataset = TrajectoryGeoDataset(self.val_data_path, self.history_len, self.future_len, self.edge_features, device=self.device)
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

import torch
from torch_geometric.data import Dataset, Data
import h5py
import numpy as np
from tqdm import tqdm
import networkx as nx

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # Specify that 'future_edge_features' should not be concatenated across the zeroth dimension
        if (key == 'y') or (key == 'history_indices') or (key == 'future_indices'):
            return None  # This will add a new batch dimension during batching
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)  # Default behaviour

class TrajectoryGeoDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, edge_features=None, device=None, embedding_dim=None):
        super().__init__()
        self.ct = 0
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.embedding_dim = embedding_dim
        self.device = device
        if 'road_type' in self.edge_features:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates, self.road_type = self.load_new_format(self.file_path, self.edge_features, self.device)
            self.road_type = torch.tensor(self.road_type, dtype=torch.float64, device=self.device)
            self.num_road_types = self.road_type.size(1)
        else:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(self.file_path, self.edge_features, self.device)
        
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float64, device=self.device)
        self.edge_index = self._build_edge_index()

    @staticmethod
    def load_new_format(file_path, edge_features, device):
        paths = []
        with h5py.File(file_path, 'r') as new_hf:
            node_coordinates = torch.tensor(new_hf['graph']['node_coordinates'][:], dtype=torch.float, device=device)
            # Normalize the coordinates to (0, 1) if any of the coordinates is larger than 1
            if node_coordinates.max() > 1:
                max_values = node_coordinates.max(0)[0]
                min_values = node_coordinates.min(0)[0]
                node_coordinates[:, 0] = (node_coordinates[:, 0] - min_values[0]) / (max_values[0] - min_values[0])
                node_coordinates[:, 1] = (node_coordinates[:, 1] - min_values[1]) / (max_values[1] - min_values[1])
            edges = new_hf['graph']['edges'][:]
            edge_coordinates = node_coordinates[edges]
            nodes = [(i, {'pos': torch.tensor(pos, device=device)}) for i, pos in enumerate(node_coordinates)]
            edges = [tuple(edge) for edge in edges]
            for i in tqdm(new_hf['trajectories'].keys()):
                path_group = new_hf['trajectories'][i]
                path = {attr: torch.tensor(path_group[attr][()], device=device) for attr in path_group.keys() if attr in ['coordinates', 'edge_idxs', 'edge_orientations']}
                paths.append(path)
            if 'road_type' in edge_features:
                onehot_encoded_road_type = new_hf['graph']['road_type'][:]
                return paths, nodes, edges, edge_coordinates, onehot_encoded_road_type
            else:
                return paths, nodes, edges, edge_coordinates
    
    def _build_edge_index(self):
        print("Building edge index for line graph...")
        self.G = self.build_graph()
        self.num_edges = self.G.number_of_edges()
        edge_index = torch.tensor([[e[0], e[1]] for e in self.G.edges(data=True)], dtype=torch.long).t().contiguous()
        edge_to_index = {tuple(e[:2]): e[2]['index'] for e in self.G.edges(data=True)}
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

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        edge_idxs = trajectory['edge_idxs']
        if 'edge_orientations' in self.edge_features:
            edge_orientations = trajectory['edge_orientations']
        
        padding_length = max(self.history_len + self.future_len - len(edge_idxs), 0)
        edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)

        # Split into history and future
        history_indices = edge_idxs[:self.history_len]
        future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]
        future_indices_check = future_indices[future_indices >= 0]
        
        # Check if datapoint is valid
        # 1. Check if history and future are connected
        subgraph_nodes = set()
        for idx in torch.cat((history_indices, future_indices_check), dim=0):
            edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
            subgraph_nodes.update(edge)
        # Create a subgraph with these nodes
        subgraph = self.G.subgraph(subgraph_nodes)
        connected = nx.is_connected(subgraph)
        
        # 2. Check if neither history nor future contain loops
        if len(future_indices_check) <= 1:
            acyclic_fut = True
        else:
            subgraph_nodes = []
            subgraph_edges = []
            for idx in future_indices_check:
                edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
                subgraph_nodes.append(edge)
                subgraph_edges.append(edge)
            subgraph = nx.Graph()
            subgraph.add_edges_from(subgraph_edges)
            has_cycle = nx.cycle_basis(subgraph)
            acyclic_fut = len(has_cycle) == 0
        
        subgraph_nodes = []
        subgraph_edges = []
        for idx in history_indices:
            edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
            subgraph_nodes.append(edge)
            subgraph_edges.append(edge)
        subgraph = nx.Graph()
        subgraph.add_edges_from(subgraph_edges)
        has_cycle = nx.cycle_basis(subgraph)
        acyclic_hist = len(has_cycle) == 0
        
        # 3. Check if neither history nor future contain splits
        if len(future_indices_check) <= 1:
            no_splits_fut = True
        else:
            subgraph_nodes = set()
            subgraph_edges = []
            for idx in future_indices_check:
                edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
                subgraph_nodes.update(edge)
                subgraph_edges.append(edge)
            # Create a directed version of the subgraph to check for cycles
            subgraph = nx.Graph()
            subgraph.add_nodes_from(subgraph_nodes)
            subgraph.add_edges_from(subgraph_edges)
            if any(subgraph.degree(node) > 2 for node in subgraph.nodes()):
                no_splits_fut = False
            else:
                no_splits_fut = True
        
        subgraph_nodes = set()
        subgraph_edges = []
        for idx in history_indices:
            edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
            subgraph_nodes.update(edge)
            subgraph_edges.append(edge)
        # Create a directed version of the subgraph to check for cycles
        subgraph = nx.Graph()
        subgraph.add_nodes_from(subgraph_nodes)
        subgraph.add_edges_from(subgraph_edges)
        if any(subgraph.degree(node) > 2 for node in subgraph.nodes()):
            no_splits_hist = False
        else:
            no_splits_hist = True
        
        # 4. Check if history is connected and future is connected
        subgraph_nodes = set()
        for idx in history_indices:
            edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
            subgraph_nodes.update(edge)
        # Create a subgraph with these nodes
        subgraph = self.G.subgraph(subgraph_nodes)
        connected_hist = nx.is_connected(subgraph)
        
        if len(future_indices_check) <= 1:
            connected_fut = True
        else:
            subgraph_nodes = set()
            for idx in future_indices_check:
                edge = self.indexed_edges[idx][0]  # get the node tuple for each edge
                subgraph_nodes.update(edge)
            # Create a subgraph with these nodes
            subgraph = self.G.subgraph(subgraph_nodes)
            connected_fut = nx.is_connected(subgraph)
        
        if not (connected and acyclic_fut and acyclic_hist and no_splits_fut and no_splits_hist and connected_hist and connected_fut):
            print("Invalid trajectory, skipping...")
            if not connected:
                print("History and Future not connected")
            if not acyclic_fut:
                print("Future contains a cycle")
            if not acyclic_hist:
                print("History contains a cycle")
            if not no_splits_fut:
                print("Future contains a split")
            if not no_splits_hist:
                print("History contains a split")
            if not connected_hist:
                print("History not connected")
            if not connected_fut:
                print("Future not connected")
            print("History indices:", history_indices)
            print("Future indices:", future_indices_check)
            self.ct += 1
            self.__getitem__((idx + 1) % len(self.trajectories))
            
        # Extract and generate features
        history_edge_features, future_edge_features = self.generate_edge_features(history_indices, future_indices, self.edge_coordinates)
        data = MyData(x=history_edge_features,          # (batch_size * num_edges, num_edge_features)
                    edge_index=self.edge_index,         # (2, num_edges)
                    y=future_edge_features,             # (batch_size, num_edges, 1)
                    history_indices=history_indices,    # (batch_size, history_len)
                    future_indices=future_indices,      # (batch_size, future_len)
                    num_nodes=self.num_edges)
        
        return data

    def generate_edge_features(self, history_indices, future_indices, history_edge_orientations=None, future_edge_orientations=None):
        # Binary on/off edges
        valid_history_mask = history_indices >= 0
        valid_future_mask = future_indices >= 0
        
        history_one_hot_edges = torch.nn.functional.one_hot(history_indices[valid_history_mask], num_classes=len(self.edges))
        future_one_hot_edges = torch.nn.functional.one_hot(future_indices[valid_future_mask], num_classes=len(self.edges))
        
        # Sum across the time dimension to count occurrences of each edge
        history_one_hot_edges = history_one_hot_edges.sum(dim=0)  # (num_edges,)
        future_one_hot_edges = future_one_hot_edges.sum(dim=0)  # (num_edges,)
        
        # Basic History edge features = coordinates, binary encoding
        history_edge_features = history_one_hot_edges.view(-1, 1).float()
        future_edge_features = future_one_hot_edges.view(-1, 1).float()
        if 'coordinates' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
        if 'edge_orientations' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, history_edge_orientations.float()), dim=1)
        if 'road_type' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, self.road_type.float()), dim=1)
        if 'pw_distance' in self.edge_features:
            last_history_edge_coords = self.edge_coordinates[history_indices[-1]]
            edge_middles = (self.edge_coordinates[:, 0, :] + self.edge_coordinates[:, 1, :]) / 2
            last_history_edge_middle = (last_history_edge_coords[0, :] + last_history_edge_coords[1, :]) / 2  # Last edge in the history_indices
            distances = torch.norm(edge_middles - last_history_edge_middle, dim=1, keepdim=True)
            history_edge_features = torch.cat((history_edge_features, distances.float()), dim=1)
        if 'edge_length' in self.edge_features:
            edge_lengths = torch.norm(self.edge_coordinates[:, 0, :] - self.edge_coordinates[:, 1, :], dim=1, keepdim=True)
            history_edge_features = torch.cat((history_edge_features, edge_lengths.float()), dim=1)
        if 'edge_angles' in self.edge_features:
            start, end = self.find_trajectory_endpoints(history_indices)
            v1 = end - start
            starts = self.edge_coordinates[:, 0, :]
            ends = self.edge_coordinates[:, 1, :]
            vectors = ends - starts
            
            # Calculate the dot product of v1 with each vector
            dot_products = torch.sum(v1 * vectors, dim=1)
            
            # Calculate magnitudes of v1 and all vectors
            v1_mag = torch.norm(v1)
            vector_mags = torch.norm(vectors, dim=1)
            
            # Calculate the cosine of the angles
            cosines = dot_products / (v1_mag * vector_mags)
            cosines = cosines.unsqueeze(1)
            history_edge_features = torch.cat((history_edge_features, cosines.float()), dim=1)
            if torch.isnan(cosines).any():
                cosines = torch.nan_to_num(cosines, nan=0.0)
            
        history_edge_features = torch.cat((history_edge_features, torch.zeros_like(future_edge_features)), dim=1)
        history_edge_features = torch.nan_to_num(history_edge_features, nan=0.0)
        
        return history_edge_features, future_edge_features
    
    def find_trajectory_endpoints(self, edge_sequence):
        """
        Find the start and end points of a trajectory based on a sequence of edge indices,
        accounting for the direction and connection of edges.
        
        Args:
            edge_coordinates (torch.Tensor): Coordinates of all edges in the graph, shape (num_edges, 2, 2).
                                            Each edge is represented by two points [point1, point2].
            edge_sequence (torch.Tensor): Indices of edges forming the trajectory, shape (sequence_length).
        
        Returns:
            tuple: Start point and end point of the trajectory.
        """
        # Get the coordinates of edges in the sequence
        trajectory_edges = self.edge_coordinates[edge_sequence]
        
        # Determine the start point by checking the connection of the first edge with the second
        if torch.norm(trajectory_edges[0, 0] - trajectory_edges[1, 0]) < torch.norm(trajectory_edges[0, 1] - trajectory_edges[1, 0]):
            start_point = trajectory_edges[0, 1]  # Closer to the second edge's start
        else:
            start_point = trajectory_edges[0, 0]
        
        # Determine the end point by checking the connection of the last edge with the second to last
        if torch.norm(trajectory_edges[-1, 1] - trajectory_edges[-2, 1]) < torch.norm(trajectory_edges[-1, 0] - trajectory_edges[-2, 1]):
            end_point = trajectory_edges[-1, 0]  # Closer to the second to last edge's end
        else:
            end_point = trajectory_edges[-1, 1]
        
        return start_point, end_point
    
    def build_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        self.indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in self.indexed_edges:
            graph.add_edge(start, end, index=index, default_orientation=(start, end))
        return graph
    
def custom_collate_fn(batch):
    batch_size = len(batch)
    num_edges = batch[0].num_nodes

    x = torch.cat([data.x for data in batch], dim=0)
    y = torch.cat([data.y for data in batch], dim=0)
    history_indices = torch.cat([data.history_indices for data in batch], dim=0)
    future_indices = torch.cat([data.future_indices for data in batch], dim=0)

    edge_indices = [data.edge_index for data in batch]
    for i, edge_index in enumerate(edge_indices):
        edge_indices[i] = edge_index + i * num_edges

    edge_index = torch.cat(edge_indices, dim=1)
    num_nodes = batch_size * num_edges

    return MyData(x=x, edge_index=edge_index, y=y, history_indices=history_indices, future_indices=future_indices, num_nodes=num_nodes)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., device=None):
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
    timesteps = timesteps.to(device)
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb.to(device)

def generate_positional_encodings(indices, history_len, embedding_dim, device):
    """
    Generates positional encodings only for specified indices.

    Args:
        indices (torch.Tensor): The indices of history edges.
        history_len (int): Total positions (here, history length).
        embedding_dim (int): The dimensionality of each position encoding.
        device (torch.device): The device tensors are stored on.

    Returns:
        torch.Tensor: Positional encodings for specified indices with shape [len(indices), embedding_dim].
    """
    positions = torch.arange(history_len, dtype=torch.float32, device=device)
    # Get div term
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    div_term = div_term.to(device)

    # Compute positional encodings
    pe = torch.zeros((len(indices), embedding_dim), device=device)
    pe[:, 0::2] = torch.sin(positions[indices][:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(positions[indices][:, None] * div_term[None, :])

    return pe

class Edge_Encoder(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, edge_features, num_edge_features, num_timesteps, pos_encoding_dim):
        super(Edge_Encoder, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = num_timesteps
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        
        # Positional Encoding
        self.pos_encoding_dim = pos_encoding_dim
        if 'pos_encoding' in self.edge_features:
            self.pos_linear0 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
            self.pos_linear1 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_edge_features, self.hidden_channels, heads=self.num_heads))
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads))
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels * self.num_heads, self.condition_dim)  # To encode history to c
        if 'pos_encoding' in self.edge_features:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim + self.pos_encoding_dim,
                                        self.hidden_channels)  # To predict future edges
        else:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim,
                                            self.hidden_channels)  # To predict future edges
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, edge_index, indices=None, t=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices and edge features
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        # Edge Embedding
        if x.dim() == 3:
            x = x.squeeze(0)    # (bs, num_edges, num_edge_features) -> (num_edges, num_edge_features)
        for conv in self.convs:
            # print("Conv size", conv)
            x = F.relu(conv(x, edge_index)) # (num_edges, hidden_channels)
                    
        if mode == 'history':
            c = self.history_encoder(x) # (num_edges, condition_dim)
            if 'pos_encoding' in self.edge_features:
                pos_encoding = generate_positional_encodings(indices, self.history_len, self.pos_encoding_dim, device=x.device)
                encodings = F.silu(self.pos_linear0(pos_encoding))
                encodings = F.silu(self.pos_linear1(encodings))
                c = self.integrate_encodings(c, indices, encodings)
            
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)
            t_emb = t_emb.repeat(self.num_edges, 1) # (num_edges, time_embedding_dim)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=1) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=1) # Concatenate with condition c
            #print("x size", x.size())
            
            logits = self.future_decoder(x) # (num_edges, hidden_channels)
            #print(self.future_decoder)
            #print("Logits pre size", logits.size())
            logits = self.adjust_to_class_shape(logits) # (num_edges, num_classes=2)
            #print(self.adjust_to_class_shape)
            #print("Logits post size", logits.size())

            return logits.unsqueeze(0)  # (1, num_edges, num_classes=2)

    def integrate_encodings(self, features, indices, encodings):
        """
        Integrates positional encodings into the feature matrix.

        Args:
            features (torch.Tensor): Original feature matrix [num_edges, num_features].
            indices (torch.Tensor): Indices of edges where the encodings should be added.
            encodings (torch.Tensor): Positional encodings [len(indices), encoding_dim].

        Returns:
            torch.Tensor: Updated feature matrix with positional encodings integrated.
        """
        # Ensure that features are on the same device as encodings
        features = features.to(encodings.device)
        
        # Expand the features tensor to accommodate the positional encodings
        new_features = torch.cat([features, torch.zeros(features.size(0), encodings.size(1), device=features.device)], dim=1)
        
        # Place the positional encodings in the rows specified by indices
        new_features[indices, -encodings.size(1):] = encodings

        return new_features

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., device=None):
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
    timesteps = timesteps.to(device)
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb.to(device)

def generate_positional_encodings(indices, history_len, embedding_dim, device):
    """
    Generates positional encodings only for specified indices.

    Args:
        indices (torch.Tensor): The indices of history edges.
        history_len (int): Total positions (here, history length).
        embedding_dim (int): The dimensionality of each position encoding.
        device (torch.device): The device tensors are stored on.

    Returns:
        torch.Tensor: Positional encodings for specified indices with shape [len(indices), embedding_dim].
    """
    positions = torch.arange(history_len, dtype=torch.float32, device=device)
    # Get div term
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    div_term = div_term.to(device)

    # Compute positional encodings
    pe = torch.zeros((len(indices), embedding_dim), device=device)
    pe[:, 0::2] = torch.sin(positions[indices][:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(positions[indices][:, None] * div_term[None, :])

    return pe

class Edge_Encoder_MLP(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, edge_features, num_edge_features, num_timesteps, pos_encoding_dim):
        super(Edge_Encoder_MLP, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.edge_features = edge_features
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.num_classes = num_classes
        
        # Time embedding
        self.max_time = num_timesteps
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        
        # Positional Encoding
        self.pos_encoding_dim = pos_encoding_dim
        if 'pos_encoding' in self.edge_features:
            self.pos_linear0 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
            self.pos_linear1 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_layers = self.config['num_layers']
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(self.num_edge_features, self.hidden_channels))
        for _ in range(1, self.num_layers):
            self.lin_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels, self.condition_dim)  # To encode history to c
        if 'pos_encoding' in self.edge_features:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim + self.pos_encoding_dim,
                                        self.hidden_channels)  # To predict future edges
        else:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim,
                                            self.hidden_channels)  # To predict future edges
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, indices=None, t=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor
        """    
        # GNN forward pass
        
        # Edge Embedding        
        for layer in self.lin_layers:
            x = F.relu(layer(x))

        if mode == 'history':
            c = self.history_encoder(x) # (bs, num_edges, condition_dim)
            if 'pos_encoding' in self.edge_features:
                pos_encoding = generate_positional_encodings(indices, self.history_len, self.pos_encoding_dim, device=x.device)
                encodings = F.silu(self.pos_linear0(pos_encoding))
                encodings = F.silu(self.pos_linear1(encodings))
                c = self.integrate_encodings(c, indices, encodings)
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)   # (bs, time_embedding_dim)
            t_emb = t_emb.unsqueeze(1).repeat(1, x.size(1), 1) # (bs, num_edges, time_embedding_dim)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=2) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=2) # Concatenate with condition c, (bs, num_edges, hidden_channels + condition_dim + time_embedding_dim)
            
            logits = self.future_decoder(x) # (bs, num_edges, hidden_channels)
            logits = self.adjust_to_class_shape(logits) # (bs, num_edges, num_classes=2)

            return logits
        
    def integrate_encodings(self, features, indices, encodings):
        """
        Integrates positional encodings into the feature matrix.

        Args:
            features (torch.Tensor): Original feature matrix [num_edges, num_features].
            indices (torch.Tensor): Indices of edges where the encodings should be added.
            encodings (torch.Tensor): Positional encodings [len(indices), encoding_dim].

        Returns:
            torch.Tensor: Updated feature matrix with positional encodings integrated.
        """
        # Ensure that features are on the same device as encodings
        features = features.to(encodings.device)
        
        # Expand the features tensor to accommodate the positional encodings
        new_features = torch.cat([features, torch.zeros(features.size(0), encodings.size(1), device=features.device)], dim=1)
        
        # Place the positional encodings in the rows specified by indices
        new_features[indices, -encodings.size(1):] = encodings

        return new_features

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., device=None):
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
    timesteps = timesteps.to(device)
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb.to(device)
    
def generate_positional_encodings(history_len, embedding_dim, device, n=1000):
    PE = torch.zeros((history_len, embedding_dim), device=device)
    for k in range(history_len):
        for i in torch.arange(int(embedding_dim/2)):
            denominator = torch.pow(n, 2*i/embedding_dim)
            PE[k, 2*i] = torch.sin(k/denominator)
            PE[k, 2*i+1] = torch.cos(k/denominator)
    return PE

class Edge_Encoder_Residual(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, edge_features, num_edge_features, num_timesteps, pos_encoding_dim):
        super(Edge_Encoder_Residual, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = num_timesteps
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        
        # Positional Encoding
        self.pos_encoding_dim = pos_encoding_dim
        if 'pos_encoding' in self.edge_features:
            self.pos_linear0 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
            self.pos_linear1 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.theta = self.config['theta']
        
        self.convs = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        if 'pos_encoding' in self.edge_features:
            self.convs.append(GATv2Conv(self.num_edge_features + self.pos_encoding_dim + self.time_embedding_dim, self.hidden_channels, heads=self.num_heads))
            self.res_layers.append(nn.Linear(self.num_edge_features + self.pos_encoding_dim + self.time_embedding_dim, self.hidden_channels * self.num_heads))
        else: 
            self.convs.append(GATv2Conv(self.num_edge_features + self.time_embedding_dim, self.hidden_channels, heads=self.num_heads))
            self.res_layers.append(nn.Linear(self.num_edge_features + self.time_embedding_dim, self.hidden_channels * self.num_heads))
            
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, bias=False))
            self.res_layers.append(nn.Linear(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads))

        # Output layers for each task
        self.future_decoder = nn.Linear(self.hidden_channels * self.num_heads, self.hidden_channels)  # To predict future edges
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, edge_index, t, indices=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: edge attributes, including binary encoding fo edges present/absent in trajectory
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        # Edge Embedding
        batch_size = x.size(0) // self.num_edges
        if 'pos_encoding' in self.edge_features:
            pos_encoding = generate_positional_encodings(self.history_len, self.pos_encoding_dim, device=x.device)
            encodings = F.silu(self.pos_linear0(pos_encoding))
            encodings = F.silu(self.pos_linear1(encodings))
            x = self.integrate_encodings(x, indices, encodings) # (batch_size * num_edges, num_edge_features + pos_encoding_dim)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
        t_emb = self.time_linear0(t_emb)
        t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
        t_emb = self.time_linear1(t_emb)
        t_emb = F.silu(t_emb)   # (batch_size, time_embedding_dim)
        t_emb = torch.repeat_interleave(t_emb, self.num_edges, dim=0) # (batch_size * num_edges, time_embedding_dim)
        
        x = torch.cat((x, t_emb), dim=1) # Concatenate with time embedding, (batch_size * num_edges, num_edge_features + time_embedding_dim (+ pos_enc_dim))
        
        for conv, res_layer in zip(self.convs, self.res_layers):
            res = F.relu(res_layer(x))
            x = F.relu(conv(x, edge_index))
            x = self.theta * x + res        # (batch_size * num_edges, hidden_channels * num_heads)        

        logits = self.future_decoder(x) # (batch_size * num_edges, hidden_channels)
        logits = self.adjust_to_class_shape(logits) # (batch_size * num_edges, num_classes=2)

        return logits.view(batch_size, self.num_edges, -1)  # (batch_size, num_edges, num_classes=2)

    def integrate_encodings(self, features, indices, encodings):
        """
        Integrates positional encodings into the feature matrix.

        Args:
            features (torch.Tensor): Original feature matrix [num_edges, num_features].
            indices (torch.Tensor): Indices of edges where the encodings should be added.
            encodings (torch.Tensor): Positional encodings [len(indices), encoding_dim].

        Returns:
            torch.Tensor: Updated feature matrix with positional encodings integrated.
        """
        
        # Ensure that features are on the same device as encodings
        batch_size = indices.shape[0]
        encodings = encodings.repeat(batch_size, 1)
        # Ensure that features and encodings are on the same device
        features = features.to(encodings.device)
        
        # Expand the features tensor to accommodate the positional encodings
        new_features = torch.cat([features, torch.zeros(features.size(0), encodings.size(1), device=features.device)], dim=1)
        
        # Calculate batch offsets
        batch_offsets = torch.arange(batch_size, device=features.device) * self.num_edges
        
        # Flatten indices for direct access, adjust with batch offsets
        flat_indices = (indices + batch_offsets.unsqueeze(1)).flatten()
        
        # Place the positional encodings in the correct rows across all batches
        new_features[flat_indices, -encodings.size(1):] = encodings

        return new_features

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
import time
import torch.nn as nn
import torch.nn.functional as F

def make_diffusion(diffusion_config, model_config, num_edges, future_len, edge_features, device):
    """HParams -> diffusion object."""
    return CategoricalDiffusion(
        betas=get_diffusion_betas(diffusion_config, device),
        model_prediction=model_config['model_prediction'],
        model_output=model_config['model_output'],
        transition_mat_type=model_config['transition_mat_type'],
        transition_bands=model_config['transition_bands'],
        loss_type=model_config['loss_type'],
        hybrid_coeff=model_config['hybrid_coeff'],
        num_edges=num_edges,
        model_name=model_config['name'],
        future_len=future_len,
        edge_features=edge_features,
        device=device
)


def get_diffusion_betas(spec, device):
    """Get betas from the hyperparameters."""
    
    if spec['type'] == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec['start'], spec['stop'], spec['num_timesteps']).to(device)
    
    elif spec['type'] == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        def cosine_beta_schedule(timesteps, s=0.008):
            """
            Cosine schedule as described in https://arxiv.org/abs/2102.09672.

            Parameters:
            - timesteps: int, the number of timesteps for the schedule.
            - s: float, small constant to prevent numerical issues.

            Returns:
            - betas: torch.Tensor, beta values for each timestep.
            - alphas: torch.Tensor, alpha values for each timestep.
            - alpha_bars: torch.Tensor, cumulative product of alphas for each timestep.
            """
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            alphas = 1 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)

            return betas, alphas, alpha_bars
        betas, alphas, alpha_bars = cosine_beta_schedule(spec['num_timesteps'])
        return betas.to(device)
    
    elif spec['type'] == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec['num_timesteps'], 1, spec['num_timesteps']).to(device)
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
               num_edges, torch_dtype=torch.float32, model_name=None, future_len=None, edge_features=None, device=None):

        self.model_prediction = model_prediction  # *x_start*, xprev
        self.model_output = model_output  # logits or *logistic_pars*
        self.loss_type = loss_type  # kl, *hybrid*, cross_entropy_x_start
        self.hybrid_coeff = hybrid_coeff
        self.torch_dtype = torch_dtype
        self.model_name = model_name
        self.device = device

        # Data \in {0, ..., num_edges-1}
        self.num_classes = 2 # 0 or 1
        self.num_edges = num_edges
        self.future_len = future_len
        self.edge_features = edge_features
        # self.class_weights = torch.tensor([self.future_len / self.num_edges, 1 - self.future_len / self.num_edges], dtype=torch.float64)
        self.class_weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
        self.class_probs = torch.tensor([1 - self.future_len / self.num_edges, self.future_len / self.num_edges], dtype=torch.float64)
        self.transition_bands = transition_bands
        self.transition_mat_type = transition_mat_type
        self.eps = 1.e-6

        if not isinstance(betas, torch.Tensor):
            raise ValueError('expected betas to be a torch tensor')
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError('betas must be in (0, 1]')

        # Computations here in float64 for accuracy
        self.betas = betas.to(dtype=torch.float64).to(self.device, non_blocking=True)
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
        elif self.transition_mat_type == 'custom':
            q_one_step_mats = [self._get_custom_transition_mat(t)
                               for t in range(0, self.num_timesteps)]
        else:
            raise ValueError(
                f"transition_mat_type must be 'gaussian', 'uniform', 'absorbing', 'marginal_prior'"
                f", but is {self.transition_mat_type}"
                )

        self.q_onestep_mats = torch.stack(q_one_step_mats, axis=0).to(self.device, non_blocking=True)
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
                        dtype=torch.float64).to(self.device, non_blocking=True)

        # Make the values correspond to a similar type of gaussian as in the
        # gaussian diffusion case for continuous state spaces.
        values = torch.linspace(torch.tensor(0.), torch.tensor(self.num_classes-1), self.num_classes, dtype=torch.float64).to(self.device, non_blocking=True)
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
            off_diag = torch.full((self.num_classes - k,), values[k], dtype=torch.float64).to(self.device, non_blocking=True)

            mat += torch.diag(off_diag, k)
            mat += torch.diag(off_diag, -k)

        # Add diagonal values such that rows and columns sum to one.
        # Technically only the ROWS need to sum to one
        # NOTE: this normalization leads to a doubly stochastic matrix,
        # which is necessary if we want to have a uniform stationary distribution.
        diag = 1. - mat.sum(dim=1)
        mat += torch.diag_embed(diag)

        return mat.to(self.device, non_blocking=True)

    def _get_absorbing_transition_mat(self, t):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Has an absorbing state for pixelvalues self.num_classes//2.

        Args:
        t: timestep. integer scalar.

        Returns:
        Q_t: transition matrix. shape = (num_classes, num_classes).
        """
        beta_t = self.betas[t]

        diag = torch.full((self.num_classes,), 1. - beta_t, dtype=torch.float64).to(self.device, non_blocking=True)
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
        mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64).to(self.device, non_blocking=True)

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:
                    mat[i, j] = beta_t * self.class_probs[j]
                else:
                    mat[i, j] = 1 - beta_t + beta_t * self.class_probs[j]
        
        return mat
    
    def _get_custom_transition_mat(self, t):
        """
        Generates a 2x2 transition matrix for a discrete diffusion process at timestep t.

        Parameters:
        - t: int, current timestep.
        - device: torch.device, the device to place the tensor on.

        Returns:
        - transition_matrix: torch.Tensor, a 2x2 transition matrix.
        """
        alphas = 1 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bar_t = alpha_bars[t].to(self.device)
        
        # Compute the transition probabilities
        transition_matrix = torch.tensor([[alphas[t], 1 - alphas[t]],
                                        [1 - alphas[t], alphas[t]]], device=self.device)
        
        return transition_matrix

    def _at(self, q_mat, t, x):
        """
        Extract coefficients at specified timesteps t and conditioning data x in PyTorch.

        Args:
        a: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of int32 or int64 type.
            (Noisy) data. Should not be of one-hot representation, but have integer
            values representing the class values.

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
        # x.shape = (bs, num_edges) 
        # t_broadcast_shape = (bs, 1)
        # a.shape = (num_timesteps, num_classes, num_classes) 
        # out.shape = (bs, num_edges, num_classes) 
        
        # Convert `a` to the desired dtype if not already
        q_mat = q_mat.type(self.torch_dtype)        

        # Prepare t for broadcasting by adding necessary singleton dimensions
        t_broadcast = t.unsqueeze(1).to(self.device, non_blocking=True)

        # Advanced indexing in PyTorch to select elements
        return q_mat[t_broadcast, x.long()].to(self.device, non_blocking=True)  # (batch_size, num_edges, 2)

    def _at_onehot(self, q_mat, t, x):
        """Extract coefficients at specified timesteps t and conditioning data x.

        Args:
        q_mat: torch.Tensor: PyTorch tensor of constants indexed by time, dtype should be pre-set.
        t: torch.Tensor: PyTorch tensor of time indices, shape = (batch_size,).
        x: torch.Tensor: PyTorch tensor of shape (bs, ...) of float32 type.
            (Noisy) data. Should be of one-hot-type representation.

        Returns:
        out: torch.tensor: output of dot(x, q_mat[t], axis=[[-1], [1]]).
            shape = (bs, num_edges, channels=1, num_classes)
        """
        q_mat = q_mat.type(self.torch_dtype)
        
        ### Final ###
        # t.shape = (bs)
        # x.shape = (bs, num_edges, num_classes)
        # q_mat[t].shape = (bs, num_classes, num_classes)
        # out.shape = (bs, num_edges, num_classes)

        q_mat_t = q_mat[t]
        out = torch.einsum('bik,bkj->bij', x, q_mat_t).to(self.device, non_blocking=True)
        
        return out.to(self.device, non_blocking=True)

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
            shape = (bs, num_edges).
        t: torch.tensor: timestep of the diffusion process, shape (bs,).
        noise: torch.tensor: uniform noise on [0, 1) used to sample noisy data.
            shape should match (*x_start.shape, num_classes).

        Returns:
        sample: torch.tensor: same shape as x_start. noisy data.
        """
        assert noise.shape == x_start.shape + (self.num_classes,)
        logits = torch.log(self.q_probs(x_start, t) + self.eps) # (bs, num_edges, num_classes)

        # To avoid numerical issues, clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).tiny, max=1.)   # (bs, num_edges, num_classes)
        gumbel_noise = -torch.log(-torch.log(noise)).to(self.device, non_blocking=True) # (bs, num_edges, num_classes)
        
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
        """Compute logits of q(x_{t-1} | x_t, x_start_tilde) in PyTorch."""
        
        if x_start_logits:
            assert x_start.shape == x_t.shape + (self.num_classes,), (x_start.shape, x_t.shape)
        else:
            assert x_start.shape == x_t.shape, (x_start.shape, x_t.shape)
        
        # fact1 = x_t * Q_t.T
        fact1 = self._at(self.transpose_q_onestep_mats, t, x_t) # (batch_size, num_edges, num_classes)
        if x_start_logits:
            # x_start represents the logits of x_start
            # F.softmax(x_start, dim=-1) represents x_0_tilde from the D3PM paper, or x_start_tilde
            # fact2 = x_start_tilde * Q_{t-1}_bar
            fact2 = self._at_onehot(self.q_mats, t-1, F.softmax(x_start, dim=-1))   # (batch_size, num_edges, num_classes)
            tzero_logits = x_start
        else:
            fact2 = self._at(self.q_mats, t-1, x_start)
            tzero_logits = torch.log(F.one_hot(x_start.to(torch.int64), num_classes=self.num_classes) + self.eps)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
        t_broadcast = t_broadcast.expand(-1, tzero_logits.size(1), tzero_logits.size(-1)).to(self.device, non_blocking=True)   # tzero_logits.size(1) = num_edges, tzero_logits.size(-1) = num_classes

        return torch.where(t_broadcast == 0, tzero_logits, out) # (bs, num_edges, num_classes)

    def p_logits(self, model_fn, x, t, edge_features=None, edge_index=None, indices=None):
        """Compute logits of p(x_{t-1} | x_t) in PyTorch.
        p(x_{t-1}|x_t) ~ sum_over_x_start_tilde(q(x_{t-1}, x_t | x_start_tilde) * p(x_start_tilde|x_t))
        
        with q(x_{t-1}, x_t | x_start_tilde) ~ q(x_{t-1} | x_t, x_start_tilde) * q(x_t | x_start_tilde)
            where q(x_{t-1} | x_t, x_start_tilde) is the q_posterior_logits

        Args:
            model_fn (function): The model function that takes input `x` and `t` and returns the model output.
            x (torch.Tensor): The input tensor of shape (batch_size, num_edges) representing the noised input at time t.
            t (torch.Tensor): The time tensor of shape (batch_size,) representing the time step.

        Returns:
            tuple: A tuple containing two tensors:
                - model_logits (torch.Tensor): The logits of p(x_{t-1} | x_t) of shape (batch_size, num_edges, num_classes).
                - pred_x_start_logits (torch.Tensor): The logits of p(x_{t-1} | x_start) of shape (batch_size, num_edges, num_classes).
        """
        assert t.shape == (x.shape[0],)
        model_output = model_fn(edge_features, edge_index, t, indices)

        if self.model_output == 'logits':
            model_logits = model_output
        else:
            raise NotImplementedError(self.model_output)

        if self.model_prediction == 'x_start':
            pred_x_start_logits = model_logits
            t_broadcast = t.unsqueeze(1).unsqueeze(2)  # Adds new dimensions: [batch_size, 1, 1]
            t_broadcast = t_broadcast.expand(-1, pred_x_start_logits.size(1), pred_x_start_logits.size(-1)).to(self.device, non_blocking=True)   # pred_x_start_logits.size(1) = num_edges, pred_x_start_logits.size(-1) = num_classes
            model_logits = torch.where(t_broadcast == 0, pred_x_start_logits,
                                       self.q_posterior_logits(x_start=pred_x_start_logits, x_t=x, t=t, x_start_logits=True))
        elif self.model_prediction == 'xprev':
            pred_x_start_logits = model_logits
            raise NotImplementedError(self.model_prediction)
        
        assert (model_logits.shape == pred_x_start_logits.shape == x.shape + (self.num_classes,))
        return model_logits, pred_x_start_logits    # (bs, num_edges, 2)
    
    # === Sampling ===

    def p_sample(self, model_fn, x, t, noise, edge_features=None, edge_index=None, indices=None):
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        # Get model logits
        model_logits, pred_x_start_logits = self.p_logits(model_fn=model_fn, x=x, t=t, edge_features=edge_features, edge_index=edge_index, indices=indices)

        assert noise.shape == model_logits.shape, noise.shape

        # For numerical precision clip the noise to a minimum value
        noise = torch.clamp(noise, min=torch.finfo(noise.dtype).eps, max=1.)
        gumbel_noise = -torch.log(-torch.log(noise))
        
        # No noise when t == 0
        if t[0] == 0:
            sample = torch.argmax(model_logits, dim=-1)
        else:
            sample = torch.argmax(model_logits + gumbel_noise, dim=-1)

        assert sample.shape == x.shape
        assert pred_x_start_logits.shape == model_logits.shape
        
        return sample, pred_x_start_logits

    def p_sample_loop(self, model_fn, shape, num_timesteps=None, return_x_init=False, edge_features=None, edge_index=None, indices=None, task=None):
        """Ancestral sampling."""
        num_edges = shape[1]
        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.transition_mat_type in ['gaussian', 'uniform', 'custom']:
            # x_init = torch.randint(0, self.num_classes, size=shape, device=device)
            prob_class_1 = 0.5
            x_init = torch.bernoulli(torch.full(size=shape, fill_value=prob_class_1, device=device))
        elif self.transition_mat_type == 'marginal_prior':
            prob_class_1 = self.future_len / self.num_edges
            x_init = torch.bernoulli(torch.full(size=shape, fill_value=prob_class_1, device=device))
        elif self.transition_mat_type == 'absorbing':
            x_init = torch.full(shape, fill_value=self.num_classes // 2, dtype=torch.int32, device=device)
        else:
            raise ValueError(f"Invalid transition_mat_type {self.transition_mat_type}")
        
        x = x_init.clone()  # (bs, num_edges)
        edge_attr = x_init.float()
        new_edge_features = edge_features.clone()
        new_edge_features[:, -1] = edge_attr.flatten()
        if 'num_pred_edges' in self.edge_features:
            new_edge_features = torch.cat((new_edge_features, torch.zeros((new_edge_features.size(0), 1), device=new_edge_features.device)), dim=1)
        
        for i in range(num_timesteps):
            t = torch.full([shape[0]], self.num_timesteps - 1 - i, dtype=torch.long, device=device)
            noise = torch.rand(x.shape + (self.num_classes,), device=device, dtype=torch.float32)
            x, pred_x_start_logits = self.p_sample(model_fn=model_fn, x=x, t=t, noise=noise, edge_features=new_edge_features, edge_index=edge_index, indices=indices)
            if 'num_pred_edges' in self.edge_features:
                new_edge_features[:, -2] = x.flatten().float()
                new_edge_features[:, -1] = torch.sum(x, dim=1).repeat_interleave(self.num_edges)
            else:
                new_edge_features[:, -1] = x.flatten().float()

        if return_x_init:
            return x_init, x
        else:
            return x    # (val_bs, num_edges)

  # === Log likelihood / loss calculation ===
        
    def cross_entropy_x_start(self, x_start, pred_x_start_logits):
        """Calculate binary weighted cross entropy between x_start and predicted x_start logits.

        Args:
            x_start (torch.Tensor): original clean data, expected binary labels (0 or 1), shape (bs, num_edges)
            pred_x_start_logits (torch.Tensor): logits as predicted by the model

        Returns:
            torch.Tensor: scalar tensor representing the mean binary weighted cross entropy loss.
        """
        # Calculate binary cross-entropy with logits
        x_start = x_start.long().to(self.device, non_blocking=True)
        pred_x_start_logits = pred_x_start_logits.permute(0, 2, 1).float()          # (bs, num_edges, num_classes) -> (bs, num_classes, num_edges)
        pred_x_start_logits = pred_x_start_logits.transpose(1, 2).reshape(-1, 2)    # (bs*num_edges, num_classes)
        x_start = x_start.reshape(-1)                                               # (bs*num_edges)

        ce = F.cross_entropy(pred_x_start_logits, x_start, weight=self.class_weights.float().to(self.device, non_blocking=True), reduction='mean')

        return ce

    def training_losses(self, model_fn, x_start, edge_features, edge_index, indices=None):
        """Training loss calculation."""
        # Add noise to data
        batch_size, num_edges = x_start.shape
        noise = torch.rand(x_start.shape + (self.num_classes,), dtype=torch.float32)
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],))

        # t starts at zero. so x_0 is the first noisy datapoint, not the datapoint itself.
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)  # (bs, num_edges)
        
        # Replace true future with noised future
        x_t = x_t.float()   # (bs, num_edges)
        new_edge_features = edge_features.clone()
        if 'num_pred_edges' in self.edge_features:
            new_edge_features = torch.cat((new_edge_features, torch.zeros((new_edge_features.size(0), 1), device=new_edge_features.device)), dim=1)
        for i in range(x_t.shape[0]):
            if 'num_pred_edges' in self.edge_features:
                new_edge_features[i * num_edges:(i + 1)*num_edges, -2] = x_t[i]
                sum_x_t = torch.sum(x_t[i]).repeat(self.num_edges)
                new_edge_features[i * num_edges:(i + 1)*num_edges, -1] = sum_x_t
            else:
                new_edge_features[i * num_edges:(i + 1)*num_edges, -1] = x_t[i]
            
        # Calculate the loss
        if self.loss_type == 'kl':
            losses, pred_x_start_logits = self.vb_terms_bpd(model_fn=model_fn, x_start=x_start, x_t=x_t, t=t,
                                                               edge_features=new_edge_features, edge_index=edge_index)
            
            pred_x_start_logits = pred_x_start_logits.squeeze(2)    # (bs, num_edges, channels, classes) -> (bs, num_edges, classes)
            pred_x_start_logits = pred_x_start_logits.squeeze(0)    # (bs, num_edges, classes) -> (num_edges, classes)
            pred = pred_x_start_logits.argmax(dim=1)                # (num_edges, classes) -> (num_edges,)
            
            return losses, pred
            
        elif self.loss_type == 'cross_entropy_x_start':
            
            model_logits, pred_x_start_logits = self.p_logits(model_fn, x=x_t, t=t, edge_features=new_edge_features, edge_index=edge_index, indices=indices)
            losses = self.cross_entropy_x_start(x_start=x_start, pred_x_start_logits=pred_x_start_logits)
            
            pred = pred_x_start_logits.argmax(dim=2)    # (batch_size, num_edges, num_classes) -> (batch_size, num_edges)
            
            return losses, pred
            
        else:
            raise NotImplementedError(self.loss_type)

        return losses
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

    
data_config = {"dataset": "synthetic_20_traj",
    "train_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/munich_val.h5',
    "val_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/munich_val.h5',
    "history_len": 5,
    "future_len": 5,
    "num_classes": 2,
    "edge_features": ['one_hot_edges', 'coordinates', 'pos_encoding', 'pw_distance', 'edge_length', 'edge_angles'] # , 'road_type'
    }

diffusion_config = {"type": 'cosine', # Options: 'linear', 'cosine', 'jsd'
    "start": 1e-5,  # 1e-5 custom, 0.01 prior
    "stop": 0.09,  # 0.09 custom, 0.5 prior
    "num_timesteps": 100}

model_config = {"name": "edge_encoder_residual",
    "hidden_channels": 64,
    "time_embedding_dim": 16,
    "condition_dim": 32,
    "pos_encoding_dim": 16,
    "num_heads": 3,
    "num_layers": 3,
    "theta": 1.0, # controls strength of conv layers in residual model
    "dropout": 0.1,
    "model_output": "logits",
    "model_prediction": "x_start",  # Options: 'x_start','xprev'
    "transition_mat_type": 'marginal_prior',  # Options: 'gaussian', 'uniform', 'absorbing', 'marginal_prior', 'custom'
    "transition_bands": 0,
    "loss_type": "cross_entropy_x_start",  # Options: kl, cross_entropy_x_start, hybrid
    "hybrid_coeff": 0.001,  # Only used for hybrid loss type.
    }

train_config = {"batch_size": 8,
                "pretrain": False,
    "optimizer": "adam",
    "lr": 0.009,
    "gradient_accumulation": False,
    "gradient_accumulation_steps": 16,
    "num_epochs": 100,
    "learning_rate_warmup_steps": 80, # previously 10000
    "lr_decay": 0.999, # previously 0.9999
    "log_loss_every_steps": 1,
    "log_metrics_every_steps": 2,
    "save_model": False,
    "save_model_every_steps": 5}

test_config = {"batch_size": 6,
    "model_path": '/ceph/hdd/students/schmitj/experiments/synthetic_d3pm_test/synthetic_d3pm_test_edge_encoder_residual__hist5_fut5_marginal_prior_cosine_hidden_dim_64_time_dim_16_condition_dim_32.pth',
    "number_samples": 10,
    "eval_every_steps": 1
  }

wandb_config = {"exp_name": "synthetic_d3pm_test",
                "run_name": "test_sum_pred_edges",
    "project": "trajectory_prediction_using_denoising_diffusion_models",
    "entity": "joeschmit99",
    "job_type": "test",
    "notes": "",
    "tags": ["synthetic", "edge_encoder"]} 

if model_config["name"] == 'edge_encoder':
    encoder_model = Edge_Encoder
elif model_config["name"] == 'edge_encoder_mlp':
    encoder_model = Edge_Encoder_MLP
elif model_config["name"] == 'edge_encoder_residual':
    encoder_model = Edge_Encoder_Residual

model = Graph_Diffusion_Model(data_config, diffusion_config, model_config, train_config, test_config, wandb_config, encoder_model).to(device)
model.train()
model_path = test_config["model_path"]
sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = model.get_samples(load_model=False, model_path=model_path, task='predict')
torch.save(sample_list, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/test/samples_raw.pth')
fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, sample_list, valid_ids, ground_truth_hist, ground_truth_fut = model.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, return_samples=True)
torch.save(sample_list, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/test/samples.pth')
torch.save(valid_ids, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/test/valid_ids.pth')
torch.save(ground_truth_hist, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/test/gt_hist.pth')
torch.save(ground_truth_fut, '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/test/gt_fut.pth')
print("ground_truth_hist", ground_truth_hist)
print("ground_truth_fut", ground_truth_fut)
print("valid sample list", sample_list)
print("valid_ids", valid_ids)
valid_ids = [item for sublist in valid_ids for item in sublist]
print("Avg. sample time", sum(valid_ids) / len(valid_ids))
print("Val F1 Score", f1)
wandb.log({"Val F1 Score, mult samples": f1})
print("\n")
print("Val Accuracy", acc)
wandb.log({"Val Accuracy, mult samples": acc})
print("\n")
print("Val TPR", tpr)
wandb.log({"Val TPR, mult samples": tpr})
print("Average sample length", avg_sample_length)
print("\n")
wandb.log({"Val Avg. Sample length, mult samples": avg_sample_length})
print("Val Future ratio", fut_ratio)
wandb.log({"Val Future ratio, mult samples": fut_ratio})
print("\n")


# model.visualize_predictions(sample_list, ground_truth_hist, ground_truth_fut)
