import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import F1Score
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from dataset.trajectory_dataset_geometric import TrajectoryGeoDataset, custom_collate_fn
from .d3pm_diffusion import make_diffusion
from tqdm import tqdm
import logging
import os
import time
import wandb
import networkx as nx


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
        dif = make_diffusion(self.diffusion_config, self.model_config, num_edges=self.num_edges, future_len=self.future_len, edge_features=self.edge_features, device=self.device, avg_future_len=self.avg_future_len)
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
                print("Number of invalid paths in dataset:", self.train_dataset.ct // (epoch + 1))
                print("Evaluating on validation set...")
                sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary = self.get_samples(task='predict', number_samples=1)
                val_fut_ratio, val_f1, val_acc, val_tpr, val_avg_sample_length, val_valid_sample_ratio, ade, fde = self.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=1)
                print("Validation F1 Score:", val_f1)
                print("Validation Accuracy:", val_acc)
                print("Validation TPR:", val_tpr)
                print("Validation Future ratio:", val_fut_ratio)
                print("Validation Average sample length:", val_avg_sample_length)
                print("Validation Average valid sample ratio:", val_valid_sample_ratio)
                print("Validation ADE:", ade)
                print("Validation FDE:", fde)
                
                wandb.log({"Epoch": epoch, "Validation F1 Score": val_f1})
                wandb.log({"Epoch": epoch, "Validation Accuracy": val_acc})
                wandb.log({"Epoch": epoch, "Validation TPR": val_tpr})
                wandb.log({"Epoch": epoch, "Validation Future ratio": val_fut_ratio})
                wandb.log({"Epoch": epoch, "Validation Average sample length": round(val_avg_sample_length, 3)})
                wandb.log({"Epoch": epoch, "Validation Average valid sample ratio": round(val_valid_sample_ratio, 3)})
                wandb.log({"Epoch": epoch, "Validation ADE": ade})
                wandb.log({"Epoch": epoch, "Validation FDE": fde})
                        
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
                                                 device=self.device,
                                                 avg_future_len=self.avg_future_len).p_sample_loop(model_fn=model_fn,
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
                                                    device=self.device,
                                                    avg_future_len=self.avg_future_len).p_sample_loop(model_fn=model_fn,
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
                features = ''
                for feature in self.edge_features:
                    features += feature + '_'
                save_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/' + f'{self.wandb_config['exp_name']}' + '/' + f'{self.model_config['transition_mat_type']}' + '_' + f'{self.diffusion_config['type']}/'
                if number_samples == 1:
                    torch.save(sample_list, save_path + f'samples_one_shot_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
                    print(f"Samples saved at {save_path}samples_one_shot_{features}hist{self.history_len}_fut_{self.future_len}.pth)!")
                    fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, ade, fde = self.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=number_samples)
                    wandb.log({"One-Shot F1 Score": f1})
                    wandb.log({"One-Shot Accuracy": acc})
                    wandb.log({"One-Shot TPR": tpr})
                    wandb.log({"One-Shot Future ratio": fut_ratio})
                    wandb.log({"One-Shot Average sample length": avg_sample_length})
                    wandb.log({"One-Shot Valid sample ratio": valid_sample_ratio})
                    wandb.log({"One-Shot ADE": ade})
                    wandb.log({"One-Shot FDE": fde})
                elif number_samples > 1:
                    torch.save(sample_list, save_path + f'samples_raw_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
                    fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, valid_samples, valid_ids, ade, fde, best_f1, best_tpr, best_ade, best_fde = self.eval(sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=number_samples)
                    wandb.log({"Valid Sample F1 Score": f1})
                    wandb.log({"Valid Sample Accuracy": acc})
                    wandb.log({"Valid Sample TPR": tpr})
                    wandb.log({"Valid Sample Future ratio": fut_ratio})
                    wandb.log({"Valid Sample Average sample length": avg_sample_length})
                    wandb.log({"Valid Sample Valid sample ratio": valid_sample_ratio})
                    wandb.log({"Valid Sample ADE": ade})
                    wandb.log({"Valid Sample FDE": fde})
                    wandb.log({"Best Sample F1 Score": best_f1})
                    wandb.log({"Best Sample TPR": best_tpr})
                    wandb.log({"Best Sample ADE": best_ade})
                    wandb.log({"Best Sample FDE": best_fde})
                    torch.save(valid_samples, save_path + f'samples_valid_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
                    torch.save(valid_ids, save_path + f'valid_ids_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
                    
                torch.save(ground_truth_hist, save_path + f'gt_hist_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
                torch.save(ground_truth_fut, save_path + f'gt_fut_' + features + f'hist{self.history_len}_fut_{self.future_len}.pth')
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

    def eval(self, sample_binary_list, sample_list, ground_truth_hist, ground_truth_fut, ground_truth_fut_binary, number_samples=None, return_samples=False):
        """
        Evaluate the model's performance.

        :param sample_list: A list of predicted edge indices.
        :param ground_truth_hist: A list of actual history edge indices.
        :param ground_truth_fut: A list of actual future edge indices.
        """
        if number_samples is None:
            number_samples = self.test_config['number_samples']
        
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
                    edges = [self.train_dataset.indexed_edges[idx][0] for idx in edge_indices]  # Adjust this if your graph structure differs

                    # Create a subgraph from these edges
                    subgraph = nx.Graph()
                    subgraph.add_edges_from(edges)
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
        
        def get_valid_samples(sample_list, number_samples):
            valid_sample_ratio = 0
            if number_samples > 1:
                valid_samples = []
                binary_valid_samples = []
                valid_ids = []
                valid_ct = 0
                for i in range(len(sample_list)):
                    valid_sample_list = []
                    binary_valid_sample_list = []
                    valid_id_list = []
                    
                    transposed_data = list(zip(*sample_list[i]))
                    for j in range(len(transposed_data)):
                        preds = transposed_data[j]
                        valid_sample_len = 0
                        valid_index = None
                        binary_valid_sample = torch.zeros(self.num_edges, device='cpu')
                        valid_sample = torch.tensor([])
                        for i, sample in enumerate(preds):
                            valid, sample = check_sample(sample)
                            if valid and len(sample) > valid_sample_len:
                                valid_sample_len = len(sample)
                                valid_sample = sample
                                binary_valid_sample[valid_sample] = 1
                                valid_index = i
                        if valid_index is None:
                            random_index = torch.randint(0, len(preds), (1,)).item()
                            random_sample = preds[random_index]
                            binary_valid_sample[random_sample] = 1
                            valid_id_list.append(valid_index)
                            valid_sample_list.append(random_sample)
                            binary_valid_sample_list.append(binary_valid_sample)
                        else:
                            valid_ct += 1 / len(transposed_data)
                            valid_id_list.append(valid_index)
                            valid_sample_list.append(valid_sample)
                            binary_valid_sample_list.append(binary_valid_sample)
                    
                    binary_valid_sample_list = torch.stack(binary_valid_sample_list, dim=0)
                    valid_ids.append(valid_id_list)
                    valid_samples.append(valid_sample_list)
                    binary_valid_samples.append(binary_valid_sample_list)
                
                valid_sample_ratio = valid_ct / len(sample_list)
                return valid_samples, binary_valid_samples, valid_ids, valid_sample_ratio
            elif number_samples == 1:
                valid_ct = 0
                for sample_sublist in sample_list:
                    for sample in sample_sublist:
                        valid, sample = check_sample(sample)
                        if valid:
                            valid_ct += 1 / len(sample_sublist)
                valid_sample_ratio = valid_ct / len(self.val_dataloader)
                return valid_sample_ratio
                
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
        
        def calculate_sample_f1(tpr, prec, best=False):
            """
            Calculates the F1 score for a given list of samples and ground truth futures.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth futures.

            Returns:
                float: The F1 score.

            """
            if best:
                best_f1_scores = []
                for recall_set, precision_set in zip(tpr, prec):
                    for r_values, p_values in zip(recall_set, precision_set):
                        f1_scores = [2 * r * p / (r + p) if (r + p) != 0 else 0 for r, p in zip(r_values, p_values)]
                        best_f1_scores.append(max(f1_scores))
                return sum(best_f1_scores) / len(best_f1_scores)
            else:
                if (prec + tpr) > 0:
                    f1_score = 2 * (prec * tpr) / (prec + tpr)
                else:
                    f1_score = 0
                return f1_score
        
        def calculate_sample_accuracy(sample_binary_list, ground_truth_fut_binary, valid_id=None, valid=False):
            """
            Calculate the accuracy of the samples.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                float: The accuracy of the samples.
            """
            if valid:
                acc = []
                for i, (sample_sublist, ground_truth_sublist) in enumerate(zip(sample_binary_list, ground_truth_fut_binary)):
                    for j, (sample, ground_truth) in enumerate(zip(sample_sublist, ground_truth_sublist)):
                        if valid_id[i][j] is not None:
                            if torch.sum(sample) > 0:
                                acc.append(torch.sum(sample * ground_truth).item() / self.num_edges)
                return sum(acc) / len(acc)
            else:
                acc = 0
                for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                    ground_truth = ground_truth_sublist.flatten()
                    sample = sample_sublist.flatten()
                    acc += torch.sum(sample == ground_truth).item() / ground_truth.size(0)
                return acc / len(self.val_dataloader)
        
        def calculate_sample_tpr(sample_binary_list, ground_truth_fut_binary, valid_id=None, valid=False, best=False):
            """
            Calculate the true positive rate of the samples.

            Args:
                sample_list (list): A list of samples.
                ground_truth_fut (list): A list of ground truth future trajectories.

            Returns:
                float: The true positive rate of the samples.
            """
            if valid:
                tpr = []
                for i, (sample_sublist, ground_truth_sublist) in enumerate(zip(sample_binary_list, ground_truth_fut_binary)):
                    for j, (sample, ground_truth) in enumerate(zip(sample_sublist, ground_truth_sublist)):
                        if valid_id[i][j] is not None:
                            if torch.sum(sample) > 0:
                                tpr.append(torch.sum(sample * ground_truth).item() / torch.sum(ground_truth).item())
                return sum(tpr) / len(tpr)
            else:
                all_tprs = []
                if best:
                    tpr = []
                    for i in range(len(sample_binary_list)):
                        batch_tpr = []
                        transposed_data = list(zip(*sample_binary_list[i]))
                        for j in range(len(transposed_data)):
                            preds = transposed_data[j]
                            best_tpr = 0
                            tpr_sublist = []
                            for sample in preds:
                                if torch.sum(ground_truth_fut_binary[i][j]) > 0:
                                    tpr_int = torch.sum(sample * ground_truth_fut_binary[i][j]).item() / torch.sum(ground_truth_fut_binary[i][j]).item()
                                else:
                                    tpr_int = 0
                                
                                tpr_sublist.append(tpr_int)
                                if tpr_int > best_tpr:
                                    best_tpr = tpr_int
                            tpr.append(best_tpr)
                            batch_tpr.append(tpr_sublist)
                        all_tprs.append(batch_tpr)
                    return sum(tpr) / len(tpr), all_tprs
                else:
                    tpr = 0
                    for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                        ground_truth = ground_truth_sublist.flatten()
                        sample = sample_sublist.flatten()
                        if torch.sum(ground_truth) > 0:
                            tpr += torch.sum(sample * ground_truth).item() / torch.sum(ground_truth).item()
                    return tpr / len(self.val_dataloader)
        
        def calculate_sample_prec(sample_binary_list, ground_truth_fut_binary, valid_id=None, valid=False, best=False):
            if valid:
                prec = []
                for i, (sample_sublist, ground_truth_sublist) in enumerate(zip(sample_binary_list, ground_truth_fut_binary)):
                    for j, (sample, ground_truth) in enumerate(zip(sample_sublist, ground_truth_sublist)):
                        if valid_id[i][j] is not None:
                            if torch.sum(sample) > 0:
                                prec.append(torch.sum(sample * ground_truth).item() / torch.sum(sample).item())
                return sum(prec) / len(prec)
            else:
                all_prec = []
                if best:
                    prec = []
                    for i in range(len(sample_binary_list)):
                        batch_prec = []
                        transposed_data = list(zip(*sample_binary_list[i]))
                        for j in range(len(transposed_data)):
                            preds = transposed_data[j]
                            best_prec = 0
                            prec_sublist = []
                            for sample in preds:
                                if torch.sum(sample) > 0:
                                    prec_int = torch.sum(sample * ground_truth_fut_binary[i][j]).item() / torch.sum(sample).item()
                                else:
                                    prec_int = 0
                                
                                prec_sublist.append(prec_int)
                                if prec_int > best_prec:
                                    best_prec = prec_int
                            prec.append(best_prec)
                            batch_prec.append(prec_sublist)
                        all_prec.append(batch_prec)
                    return sum(prec) / len(prec), all_prec
                else:
                    prec = 0
                    for sample_sublist, ground_truth_sublist in zip(sample_binary_list, ground_truth_fut_binary):
                        ground_truth = ground_truth_sublist.flatten()
                        sample = sample_sublist.flatten()
                        if torch.sum(sample) > 0:
                            prec += torch.sum(sample * ground_truth).item() / torch.sum(sample).item()
                    return prec / len(self.val_dataloader)
        
        def calculate_avg_sample_length(sample_list, valid_id=None, valid=False):
            """
            Calculate the average sample length.

            Args:
                sample_list (list): A list of samples.

            Returns:
                float: The average sample length.
            """
            if valid:
                total_len = 0
                ct = 0
                for i, sample_sublist in enumerate(sample_list):
                    for j, sample in enumerate(sample_sublist):
                        if valid_id[i][j] is not None:
                            total_len += len(sample)
                            ct += 1
                return total_len / ct
            else:
                total_len = 0
                for sample_sublist in sample_list:
                    for sample in sample_sublist:
                        total_len += len(sample) / len(sample_sublist)
                return total_len / len(self.val_dataloader)
        
        def build_node_sequence(edge_sequence, indexed_edges, start_point):
            # Convert list of tuples to a tensor of shape [num_edges, 2]
            if len(edge_sequence) == 0:
                return []
            edges = [list(edge[0]) for edge in indexed_edges]
            edge_tensor = torch.tensor(edges, dtype=torch.long)[edge_sequence]
            
            # Initialize the list of nodes with the start point
            node_sequence = [start_point]
            
            # Current node to find the next connected edge
            current_node = start_point
            
            if current_node not in edge_tensor:
                return []
            # Continue until we have traversed all edges
            while len(node_sequence) < len(edge_sequence) + 1:
                for i in range(edge_tensor.size(0)):
                    # Check if the current edge connects to the current node
                    if edge_tensor[i, 0] == current_node:
                        # Append the connected node and update the current node
                        node_sequence.append(edge_tensor[i, 1].item())
                        current_node = edge_tensor[i, 1].item()
                    elif edge_tensor[i, 1] == current_node:
                        # Append the connected node and update the current node
                        node_sequence.append(edge_tensor[i, 0].item())
                        current_node = edge_tensor[i, 0].item()

            return node_sequence
        
        def find_trajectory_endpoints(edge_sequence, edge_coordinates, indexed_edges, gt_fut=None):
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
            edges = [list(edge[0]) for edge in indexed_edges]
            if len(edge_sequence) <= 1:
                edge_sequence = torch.cat((edge_sequence, torch.tensor([gt_fut[0]])), 0)
            
            # Get the coordinates of edges in the sequence
            trajectory_edges = edge_coordinates[edge_sequence]
            
            # Determine the start point by checking the connection of the first edge with the second
            if torch.norm(trajectory_edges[0, 0] - trajectory_edges[1, 0]) < torch.norm(trajectory_edges[0, 1] - trajectory_edges[1, 0]):
                start_point_coord = trajectory_edges[0, 1]  # Closer to the second edge's start
                start_point = edges[edge_sequence[0]][1]
            else:
                start_point_coord = trajectory_edges[0, 0]
                start_point = edges[edge_sequence[0]][0]
            
            # Determine the end point by checking the connection of the last edge with the second to last
            if torch.norm(trajectory_edges[-1, 1] - trajectory_edges[-2, 1]) < torch.norm(trajectory_edges[-1, 0] - trajectory_edges[-2, 1]):
                end_point_coord = trajectory_edges[-1, 0]  # Closer to the second to last edge's end
                end_point = edges[edge_sequence[-1]][0]
            else:
                end_point_coord = trajectory_edges[-1, 1]
                end_point = edges[edge_sequence[-1]][1]
            
            return start_point_coord, end_point_coord, start_point, end_point
        
        def calculate_ade_fde(batched_preds, batched_gt_futs, batched_gt_hists, edge_coordinates, indexed_edges, best=False):
            if best:
                ade_list = []
                fde_list = []
                for batch_idx in tqdm(range(len(batched_gt_futs))):
                    for idx in range(len(batched_gt_futs[batch_idx])):
                        preds = batched_preds[batch_idx]
                        gt_hist = batched_gt_hists[batch_idx][idx]
                        gt_fut_ = batched_gt_futs[batch_idx][idx]
                        gt_fut = gt_fut_[gt_fut_ != -1]
                        start_point_coords, end_point_coords, start_point, end_point = find_trajectory_endpoints(gt_hist, edge_coordinates, indexed_edges, gt_fut)
                        gt_fut_nodes = build_node_sequence(gt_fut, indexed_edges, end_point)
                        
                        best_ade = 1
                        best_fde = 1
                        for pred in preds:
                            ade = 0
                            fde = 0
                            sample = pred[idx]
                            pred_nodes = [indexed_edges[j][0] for j in sample]
                            unique_pred_nodes = set()
                            for node in pred_nodes:
                                unique_pred_nodes.update(node)  # Add both elements of the tuple to the set
                            # Convert the set back to a list
                            unique_pred_nodes = list(unique_pred_nodes)

                            if len(gt_fut) == 0 and len(sample) == 0:
                                ade = 0
                                fde = 0
                            elif len(sample) == 0 and len(gt_fut) > 0:
                                if len(gt_fut_nodes) == 0:
                                    continue
                                for i in range(len(gt_fut)):
                                    ade += torch.norm(self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i+1]][1]['pos'])
                                ade /= len(gt_fut)
                                fde = torch.norm(self.train_dataset.nodes[end_point][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                            elif len(sample) > 0 and len(gt_fut) == 0:
                                max_dist = float('-inf')
                                farthest_pred_node = None
                                for pred_node in unique_pred_nodes:
                                    ade += torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                    dist = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                    if dist > max_dist:
                                        max_dist = dist
                                        farthest_pred_node = pred_node
                                ade /= len(sample)
                                fde = torch.norm(self.train_dataset.nodes[farthest_pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                            elif len(sample) > 0 and len(gt_fut) > 0:
                                if len(gt_fut_nodes) == 0:
                                    continue
                                min_dist_fde = float('inf')
                                clostest_pred_node = None
                                for i, pred_node in enumerate(unique_pred_nodes):
                                    min_dist = float('inf')
                                    closest_gt_fut_node = None
                                    for gt_fut_node in gt_fut_nodes:
                                        dist = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[gt_fut_node][1]['pos'])
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_gt_fut_node = gt_fut_node
                                    ade += torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[closest_gt_fut_node][1]['pos'])
                                    dist_fde = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                                    if dist_fde < min_dist_fde:
                                        fde = dist_fde
                                        min_dist_fde = dist_fde
                                if len(sample) > len(gt_fut):
                                    ade /= len(sample)
                                else:
                                    ade /= len(gt_fut)
                            if ade.item() < best_ade:
                                best_ade = ade.item()
                            if fde.item() < best_fde:
                                best_fde = fde.item()
                        if len(gt_fut_nodes) > 0:
                            ade_list.append(best_ade)
                            fde_list.append(best_fde)
                return torch.mean(torch.tensor(ade_list)), torch.mean(torch.tensor(fde_list))
            
            else:
                ade_list = []
                fde_list = []
                for batch_idx in tqdm(range(len(batched_gt_futs))):
                    for idx in range(len(batched_gt_futs[batch_idx])):
                        pred = batched_preds[batch_idx][idx]
                        pred_nodes = [indexed_edges[j][0] for j in pred]
                        
                        unique_pred_nodes = set()
                        for node in pred_nodes:
                            unique_pred_nodes.update(node)  # Add both elements of the tuple to the set

                        # Convert the set back to a list
                        unique_pred_nodes = list(unique_pred_nodes)
                        gt_hist = batched_gt_hists[batch_idx][idx]
                        gt_fut_ = batched_gt_futs[batch_idx][idx]
                        gt_fut = gt_fut_[gt_fut_ != -1]
                        start_point_coords, end_point_coords, start_point, end_point = find_trajectory_endpoints(gt_hist, edge_coordinates, indexed_edges, gt_fut)
                        gt_fut_nodes = build_node_sequence(gt_fut, indexed_edges, end_point)

                        ade = 0
                        fde = 0
                        if len(gt_fut) == 0 and len(pred) == 0:
                            ade = 0
                            fde = 0
                        elif len(pred) == 0 and len(gt_fut) > 0:
                            if len(gt_fut_nodes) == 0:
                                continue
                            for i in range(len(gt_fut)):
                                ade += torch.norm(self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i+1]][1]['pos'])
                            ade /= len(gt_fut)
                            fde = torch.norm(self.train_dataset.nodes[end_point][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                        elif len(pred) > 0 and len(gt_fut) == 0:
                            max_dist = float('-inf')
                            farthest_pred_node = None
                            for pred_node in unique_pred_nodes:
                                ade += torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                dist = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                if dist > max_dist:
                                    max_dist = dist
                                    farthest_pred_node = pred_node
                            ade /= len(pred)
                            fde = torch.norm(self.train_dataset.nodes[farthest_pred_node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                        elif len(pred) > 0 and len(gt_fut) > 0:
                            if len(gt_fut_nodes) == 0:
                                continue
                            min_dist_fde = float('inf')
                            clostest_pred_node = None
                            for i, pred_node in enumerate(unique_pred_nodes):
                                min_dist = float('inf')
                                closest_gt_fut_node = None
                                for gt_fut_node in gt_fut_nodes:
                                    dist = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[gt_fut_node][1]['pos'])
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_gt_fut_node = gt_fut_node
                                ade += torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[closest_gt_fut_node][1]['pos'])
                                dist_fde = torch.norm(self.train_dataset.nodes[pred_node][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                                if dist_fde < min_dist_fde:
                                    fde = dist_fde
                                    min_dist_fde = dist_fde
                            if len(pred) > len(gt_fut):
                                ade /= len(pred)
                            else:
                                ade /= len(gt_fut)

                        ade_list.append(ade)
                        fde_list.append(fde)
                return torch.mean(torch.tensor(ade_list)), torch.mean(torch.tensor(fde_list))
        
        def calculate_ade_fde_valid(batched_preds, valid_ids, batched_gt_futs, batched_gt_hists, edge_coordinates, indexed_edges):
            ade_list = []
            fde_list = []
            for batch_idx in tqdm(range(len(batched_gt_futs))):
                for idx in range(len(batched_gt_futs[batch_idx])):
                    pred = batched_preds[batch_idx][idx]
                    pred_nodes = [indexed_edges[j][0] for j in pred]
                    valid = valid_ids[batch_idx][idx]
                    gt_hist = batched_gt_hists[batch_idx][idx]                    
                    gt_fut_ = batched_gt_futs[batch_idx][idx]
                    gt_fut = gt_fut_[gt_fut_ != -1]
                    start_point_coords, end_point_coords, start_point, end_point = find_trajectory_endpoints(gt_hist, edge_coordinates, indexed_edges, gt_fut)
                    gt_fut_nodes = build_node_sequence(gt_fut, indexed_edges, end_point)

                    ade = 0
                    fde = 0
                    
                    # No ground truth edges and no predicted edges
                    if len(gt_fut) == 0 and len(pred) == 0:
                        ade = 0
                        fde = 0
                    elif len(pred) == 0 and len(gt_fut) > 0:
                        if len(gt_fut_nodes) == 0:
                            continue
                        for i in range(len(gt_fut)):
                            ade += torch.norm(self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i+1]][1]['pos'])
                        ade /= len(gt_fut)
                        fde = torch.norm(self.train_dataset.nodes[end_point][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                    elif len(pred) > 0 and len(gt_fut) == 0:
                        if valid is not None:
                            if len(gt_fut_nodes) == 0:
                                continue
                            pred_save = pred
                            pred_length = len(pred)
                            closest_node = None
                            min_dist = float('inf')
                            for i, edge in enumerate(pred):
                                for node in pred_nodes[i]:
                                    dist = torch.norm(self.train_dataset.nodes[node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_node = node
                                        mask = pred != edge
                                        pred = pred[mask]
                            pred_nodes_sequence = build_node_sequence(pred_save, indexed_edges, closest_node)
                            for i in range(len(pred_nodes_sequence) - 1):
                                ade += torch.norm(self.train_dataset.nodes[pred_nodes_sequence[i]][1]['pos'] - self.train_dataset.nodes[pred_nodes_sequence[i+1]][1]['pos'])
                            fde = torch.norm(self.train_dataset.nodes[end_point][1]['pos'] - self.train_dataset.nodes[pred_nodes_sequence[-1]][1]['pos'])
                        else:
                            continue
                        ade /= pred_length
                    elif len(pred) > 0 and len(gt_fut) > 0:
                        if valid is not None:
                            if len(gt_fut_nodes) == 0:
                                continue
                            if len(pred) > len(gt_fut):
                                pred_save = pred
                                pred_length = len(pred)
                                closest_node = None
                                min_dist = float('inf')
                                for i, edge in enumerate(pred):
                                    for node in pred_nodes[i]:
                                        dist = torch.norm(self.train_dataset.nodes[node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_node = node
                                            mask = pred != edge
                                            pred = pred[mask]
                                pred_nodes_sequence = build_node_sequence(pred_save, indexed_edges, closest_node)
                                for i in range(len(pred_nodes_sequence)):
                                    if i < len(gt_fut_nodes):
                                        ade += torch.norm(self.train_dataset.nodes[pred_nodes_sequence[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'])
                                    else:
                                        ade += torch.norm(self.train_dataset.nodes[pred_nodes_sequence[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                                ade /= pred_length
                                fde = torch.norm(self.train_dataset.nodes[pred_nodes_sequence[-1]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                            else:
                                pred_save = pred
                                pred_length = len(pred)
                                closest_node = None
                                min_dist = float('inf')
                                for i, edge in enumerate(pred):
                                    for node in pred_nodes[i]:
                                        dist = torch.norm(self.train_dataset.nodes[node][1]['pos'] - self.train_dataset.nodes[end_point][1]['pos'])
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest_node = node
                                            mask = pred != edge
                                            pred = pred[mask]
                                pred_nodes_sequence = build_node_sequence(pred_save, indexed_edges, closest_node)
                                for i in range(len(gt_fut_nodes)):
                                    if i < len(pred_nodes_sequence):
                                        ade += torch.norm(self.train_dataset.nodes[pred_nodes_sequence[i]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'])
                                    else:
                                        ade += torch.norm(self.train_dataset.nodes[pred_nodes_sequence[-1]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[i]][1]['pos'])
                                ade /= len(gt_fut)
                                fde = torch.norm(self.train_dataset.nodes[pred_nodes_sequence[-1]][1]['pos'] - self.train_dataset.nodes[gt_fut_nodes[-1]][1]['pos'])
                        else:
                            continue
                            
                    ade_list.append(ade)
                    fde_list.append(fde)
            return torch.mean(torch.tensor(ade_list)), torch.mean(torch.tensor(fde_list))
        
        if number_samples == 1:
            valid_sample_ratio = get_valid_samples(sample_list, number_samples)
            if self.future_len > 0:
                fut_ratio = calculate_fut_ratio(sample_list, ground_truth_fut)
            else:
                fut_ratio = 0
            tpr = calculate_sample_tpr(sample_binary_list, ground_truth_fut_binary)
            prec = calculate_sample_prec(sample_binary_list, ground_truth_fut_binary)
            f1 = calculate_sample_f1(tpr, prec)
            acc = calculate_sample_accuracy(sample_binary_list, ground_truth_fut_binary)
            avg_sample_length = calculate_avg_sample_length(sample_list)
            ade, fde = calculate_ade_fde(sample_list, ground_truth_fut, ground_truth_hist, self.train_dataset.edge_coordinates, self.train_dataset.indexed_edges)
            
            return fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, ade, fde
        elif number_samples > 1:
            valid_samples, binary_valid_samples, valid_ids, valid_sample_ratio = get_valid_samples(sample_list, number_samples)
            if self.future_len > 0:
                fut_ratio = calculate_fut_ratio(valid_samples, ground_truth_fut)
            else:
                fut_ratio = 0
            tpr = calculate_sample_tpr(binary_valid_samples, ground_truth_fut_binary, valid_ids, valid=True)
            prec = calculate_sample_prec(binary_valid_samples, ground_truth_fut_binary, valid_ids, valid=True)
            f1 = calculate_sample_f1(tpr, prec)
            acc = calculate_sample_accuracy(binary_valid_samples, ground_truth_fut_binary, valid_ids, valid=True)
            avg_sample_length = calculate_avg_sample_length(valid_samples, valid_ids, valid=True)
            ade, fde = calculate_ade_fde_valid(valid_samples, valid_ids, ground_truth_fut, ground_truth_hist, self.train_dataset.edge_coordinates, self.train_dataset.indexed_edges)
            # Get best metrics out of all samples
            best_tpr, all_tpr = calculate_sample_tpr(sample_binary_list, ground_truth_fut_binary, best=True)
            best_prec, all_prec = calculate_sample_prec(sample_binary_list, ground_truth_fut_binary, best=True)
            best_f1 = calculate_sample_f1(all_tpr, all_prec, best=True)
            best_ade, best_fde = calculate_ade_fde(sample_list, ground_truth_fut, ground_truth_hist, self.train_dataset.edge_coordinates, self.train_dataset.indexed_edges, best=True)
            
            return fut_ratio, f1, acc, tpr, avg_sample_length, valid_sample_ratio, valid_samples, valid_ids, ade, fde, best_f1, best_tpr, best_ade, best_fde
    
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
        self.avg_future_len = self.train_dataset.avg_future_len
        self.G = self.train_dataset.build_graph()
        self.nodes = self.G.nodes
        self.edges = self.G.edges(data=True)
        self.num_edges = self.G.number_of_edges()
        if 'one_hot_edges' in self.edge_features:
            self.num_edge_features = 2  # Binary History and noised binary future
        else:
            self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features += 4
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features += 1
        if 'road_type' in self.edge_features:
            self.num_edge_features += self.train_dataset.num_road_types
        if 'pw_distance' in self.edge_features:
            if 'start_end' in self.edge_features:
                self.num_edge_features += 2
            else:
                self.num_edge_features += 1
        if 'edge_length' in self.edge_features:
            self.num_edge_features += 1
        if 'edge_angles' in self.edge_features:
            self.num_edge_features += 1
        if 'num_pred_edges' in self.edge_features:
            self.num_edge_features += 1
        if 'future_len' in self.edge_features:
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
