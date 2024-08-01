'''import torch
import nn as nn
import nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import LineGraph
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
import nn as nn
import nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import F1Score
import yaml
from tqdm import tqdm
import logging
import os
import time
import wandb
from torch_geometric.data import Data



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
            config={**self.data_config, **self.model_config, **self.train_config}
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
        """def get_neighbors(line_graph, edge, device):
            neighbors = []
            for e in edge:
                neighbor_set = set()
                for i in range(line_graph.edge_index.size(1)):
                    if line_graph.edge_index[0, i] == e:
                        neighbor_set.add(line_graph.edge_index[1, i].item())
                    elif line_graph.edge_index[1, i] == e:
                        neighbor_set.add(line_graph.edge_index[0, i].item())
                neighbors.append(list(neighbor_set))
            # Create a binary tensor for neighbors
            neighbors_binary = torch.zeros((len(edge), self.num_edges), dtype=torch.long, device=device)
            
            for i, n in enumerate(neighbors):
                neighbors_binary[i, n] = 1
            
            return neighbors_binary
        
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
                visited_edges = [set() for _ in range(history_edge_features.size(0))]   # keep track of visited edges, to avoid cycles
                prediction = []
                for idx in range(self.future_len):
                    future = future_edge_indices_one_hot.clone()
                    future.zero_()
                    future[:, idx] = 1
                    """for j in range(future_edge_indices.size(0)):
                        future[j, future_edge_indices[j, idx]] = 1""""""
                    
                    neighbors = get_neighbors(self.line_graph, last_history_edge, device=history_edge_features.device)
                    logits, preds = self.model(history_edge_features, neighbors)
                    
                    # Mask logits for visited edges
                    #logits = logits.clone()
                    """for i in range(len(preds)):
                        for visited_edge in visited_edges[i]:
                            logits[i, neighbors[i] == visited_edge] = float('-inf')
                    
                    # Update history and visited edges
                    for i in range(len(preds)):
                        visited_edges[i].add(preds[i].item())  # Add predicted edge to visited set
                        history_edge_features[i] = history_edge_features[i].clone()
                        history_edge_features[i, preds[i], 0] = 1"""  # Add it to history edge features
                    last_history_edge = preds   # Update last history edge
                    loss = F.binary_cross_entropy_with_logits(logits, future)
                    prediction.append(preds)
                    loss.backward()

                pred_fut.append(prediction)
                ground_truth_fut.append(future_edge_indices_one_hot)
                # Calculate the loss (cross-entropy)
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_data_loader)
            f1_score = F1Score(task='binary', average='macro', num_classes=2)
            #f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach(), torch.flatten(torch.cat(ground_truth_fut)).detach())
            # Logging
            if (epoch + 1) % self.log_loss_every_steps == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
                wandb.log({"epoch": epoch + 1, "loss": avg_loss})
                #wandb.log({"epoch": epoch + 1, "average_F1_score": f1_epoch.item()})
                print(pred_fut)
                print(future_edge_indices)
                self.log.info(f"Epoch {epoch + 1} Average Loss: {avg_loss}")
                print("Epoch:", epoch + 1)
                print("Loss:", avg_loss)
                #print("F1:", f1_epoch.item())"""
        def transform_tensor(neighbor_tensor, current_edge):
            batch_size = neighbor_tensor.size(0)
            max_neighbors = (neighbor_tensor == 1).sum(dim=1).max().item()
            transformed_tensor = torch.zeros((batch_size, max_neighbors), dtype=torch.long)

            for i in range(batch_size):
                # Find indices of neighbors
                neighbor_indices = (neighbor_tensor[i] == 1).nonzero(as_tuple=False).squeeze()
                # Only keep the index where it matches the current edge
                valid_indices = (neighbor_indices == current_edge[i]).nonzero(as_tuple=False).squeeze()
                # Set the values in the transformed tensor
                transformed_tensor[i, valid_indices] = 1

            return transformed_tensor
        
        """def group_and_pad(tensor, batch_size, pad_value=0):
            # Get unique groups
            groups = torch.unique(tensor[:, 0])
            grouped_sequences = []
            
            # Find the maximum length of sequences
            max_len = 0
            for group in groups:
                group_elements = tensor[tensor[:, 0] == group][:, 1]
                grouped_sequences.append(group_elements)
                max_len = max(max_len, len(group_elements))
            
            # Pad the sequences
            padded_sequences = []
            for seq in grouped_sequences:
                padded_seq = torch.cat([seq, torch.full((max_len - len(seq),), pad_value)])
                padded_sequences.append(padded_seq)
            
            return torch.stack(padded_sequences)"""
        
        def group_and_pad(tensor, batch_size, pad_value=0, missing_value=-1):
            grouped_sequences = []

            # Iterate through all possible groups
            for group in range(batch_size):
                group_elements = tensor[tensor[:, 0] == group][:, 1]
                if len(group_elements) == 0:
                    # If the group is missing, add the missing value
                    group_elements = torch.tensor([missing_value])
                grouped_sequences.append(group_elements)

            # Find the maximum length of sequences
            max_len = max(len(seq) for seq in grouped_sequences)
            
            # Pad the sequences
            padded_sequences = []
            for seq in grouped_sequences:
                padded_seq = torch.cat([seq, torch.full((max_len - len(seq),), pad_value)])
                padded_sequences.append(padded_seq)
            
            return torch.stack(padded_sequences)
        
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            total_loss = 0
            pred_fut = []
            ground_truth_fut = []
            for batch in self.train_data_loader:
                hidden = self.model.init_hidden(self.batch_size)
                history_edge_features = batch["history_edge_features"]
                history_edge_indices = batch["history_indices"]
                history_edge_indices_one_hot = history_edge_features[:, :, 0]
                initial_edge = batch["history_indices"][:, -1]
                future_edge_indices = batch["future_indices"]
                future_edge_features = batch["future_edge_features"]
                future_edge_indices_one_hot = future_edge_features[:, :, 0]
                batch_size_act = history_edge_features.size(0)

                self.optimizer.zero_grad()
                loss = 0
                preds = []
                preds_binary = torch.zeros_like(future_edge_indices_one_hot)
                for t in range(self.future_len):
                    if t == 0:
                        current_edge = initial_edge
                    else:
                        current_edge = future_edge_indices[:, t-1]
                    
                    true_neighbors = self.get_neighbors(self.line_graph, current_edge)
                    # TODO: Use history edge features at t == 0, then add true future_edge_features at t-1
                    input_features = self.model.get_neighbor_features(history_edge_features, true_neighbors)
                    out, hidden = self.model(input_features, hidden)
                    logits = out.squeeze(-1)  # Remove the last dimension to match (bs, num_neighbors)
                    
                    #masked_logits = logits.clone()
                    #masked_logits[true_neighbors == 0] = -100    # Mask out true neighbors
                    #masked_logits[history_edge_indices_one_hot == 1] = -100   # Mask out history
                    
                    predicted_edge_indices = torch.argmax(logits, dim=1)
                    true_neighbors_padded = group_and_pad(torch.argwhere(true_neighbors == 1), batch_size_act)  # (bs, num_neighbors)
                    
                    predicted_edges = true_neighbors_padded.gather(1, predicted_edge_indices.unsqueeze(1)).squeeze(1)
                    preds.append(predicted_edges)
                    ground_truth = transform_tensor(true_neighbors, future_edge_indices[:, t])
                    loss += criterion(logits, ground_truth.float())
                
                for i in range(self.batch_size):
                    pred_binary = torch.zeros(self.num_edges, dtype=torch.float)
                    pred_binary[torch.stack(preds).t()[i]] = 1
                    preds_binary[i] = pred_binary.clone().detach()
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pred_fut.append(preds_binary)
                ground_truth_fut.append(future_edge_indices_one_hot)
                
            preds = [torch.stack(preds).t()]
            avg_loss = total_loss / len(self.train_data_loader)
            f1_score = F1Score(task='binary', average='macro', num_classes=2)
            f1_epoch = f1_score(torch.flatten(torch.cat(pred_fut)).detach(), torch.flatten(torch.cat(ground_truth_fut)).detach())

            if (epoch + 1) % self.log_loss_every_steps == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss:.4f}, F1 Score: {f1_epoch.item()}")
                print("History Edges:", history_edge_indices)
                print("Predicted Edges:", preds)
                print("Future_edge_indices", future_edge_indices)
        
        print("> Training Complete!\n")
        
    """def get_neighbors(self, line_graph, edge):
        neighbors = []
        for e in edge:
            # When e == -1, neighbor_tensor = torch.zeros(self.num_edges, dtype=torch.long)
            neighbor_set = set()
            for i in range(line_graph.edge_index.size(1)):
                if line_graph.edge_index[0, i] == e:
                    neighbor_set.add(line_graph.edge_index[1, i].item())
                elif line_graph.edge_index[1, i] == e:
                    neighbor_set.add(line_graph.edge_index[0, i].item())
            neighbor_tensor = torch.zeros(self.num_edges, dtype=torch.long)
            neighbor_tensor[list(neighbor_set)] = 1
            neighbors.append(neighbor_tensor)
        
        return torch.stack(neighbors)"""
        
    def get_neighbors(self, line_graph, edge):
        # Ensure edge is a tensor and flatten it for ease of operations
        edge_tensor = edge.flatten()  # Flatten to ensure it is of shape (batch_size,)

        # Create a 2D boolean mask for matching edges in edge_index
        mask0 = line_graph.edge_index[0].unsqueeze(1) == edge_tensor.unsqueeze(0)
        mask1 = line_graph.edge_index[1].unsqueeze(1) == edge_tensor.unsqueeze(0)

        # Find neighbors: if mask0[i, j] is True, the neighbor is edge_index[1, i], and vice versa for mask1
        neighbors_index_0 = torch.where(mask0, line_graph.edge_index[1].unsqueeze(1), torch.full_like(line_graph.edge_index[1].unsqueeze(1), -1))
        neighbors_index_1 = torch.where(mask1, line_graph.edge_index[0].unsqueeze(1), torch.full_like(line_graph.edge_index[0].unsqueeze(1), -1))

        # Combine the neighbors from both checks and remove the placeholder -1
        combined_neighbors = torch.cat((neighbors_index_0, neighbors_index_1), dim=0)
        valid_neighbors = combined_neighbors[combined_neighbors >= 0]

        # We now need to construct the final neighbor tensor
        neighbor_tensor = torch.zeros((edge_tensor.size(0), self.num_edges), dtype=torch.long, device=edge_tensor.device)

        # Loop through each edge in batch and update the corresponding row in neighbor_tensor
        for i, e in enumerate(edge_tensor):
            # Get unique neighbor indices for the current edge e
            current_neighbors = valid_neighbors[:, i]
            unique_neighbors = torch.unique(current_neighbors[current_neighbors >= 0])
            neighbor_tensor[i, unique_neighbors] = 1

        return neighbor_tensor
        
    def save_model(self):
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_config['name'] + '_' +  self.model_config['transition_mat_type'] + '_'  + 
                                 f'_hidden_dim_{self.hidden_channels}_condition_dim_{self.condition_dim}_layers_{self.num_layers}.pth')
        torch.save(self.model.state_dict(), save_path)
        self.log.info(f"Model saved at {save_path}!")
        
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.log.info("Model loaded!\n")
    
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        def lr_lambda(epoch):
            if epoch < self.learning_rate_warmup_steps:
                return 1.0
            else:
                decay_lr = self.lr_decay_parameter ** (epoch - self.learning_rate_warmup_steps)
                return max(decay_lr, 2e-5 / self.lr)
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        print("> Optimizer and Scheduler built!\n")
        
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
        
        self.train_data_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
                
        print("> Training Dataset loaded!\n")
        
    def _build_edge_index(self):
        print("Building edge index for line graph...")
        edge_index = torch.tensor([[e[0], e[1]] for e in self.edges], dtype=torch.long).t().contiguous()
        edge_to_index = {tuple(e[:2]): e[2]['index'] for e in self.edges}
        line_graph_edges = []
        edge_list = edge_index.t().tolist()
        
        neighbor_counts = {edge_to_index[(u1, v1)]: 0 for u1, v1 in edge_list}
        
        for i, (u1, v1) in tqdm(enumerate(edge_list), total=len(edge_list), desc="Processing edges"):
            for j, (u2, v2) in enumerate(edge_list):
                if i != j and (u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2):
                    line_graph_edges.append((edge_to_index[(u1, v1)], edge_to_index[(u2, v2)]))
                    neighbor_counts[edge_to_index[(u1, v1)]] += 1

        # Create the edge index for the line graph
        edge_index = torch.tensor(line_graph_edges, dtype=torch.long).t().contiguous()
        print("> Edge index built!\n")
        
        # Find the maximum neighbor degree
        self.max_degree = max(neighbor_counts.values())
        
        return edge_index    

    def _build_test_dataloader(self):
        self.test_dataset = TrajectoryDataset(self.test_data_path, self.history_len, self.future_len, self.edge_features)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=collate_fn)
        print("> Test Dataset loaded!\n")
        
    def _build_model(self):
        #self.model = self.model(self.model_config, self.history_len, self.future_len, self.num_classes,
        #                        nodes=self.nodes, edges=self.edges,
        #                        num_edges=self.num_edges, hidden_channels=self.hidden_channels, num_edge_features=self.num_edge_features, max_degree=self.max_degree)
        self.model = self.model(self.model_config, self.num_edge_features, self.num_edges)
        print("> Model built!\n")
        
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
        
        if 'edge_indices' in new_hf['graph']:
            edge_indices = new_hf['graph']['edge_indices'][:]
            # Convert edges to a list of tuples
            # Sort edges based on their saved indices
            indexed_edges = sorted(zip(edges, edge_indices), key=lambda x: x[1])
            edges = [edge for edge, _ in indexed_edges]
        else:
            edges = [tuple(edge) for edge in edges]

        for i in tqdm(new_hf['trajectories'].keys()):
            path_group = new_hf['trajectories'][i]
            path = {attr: path_group[attr][()] for attr in path_group.keys()}
            if 'edge_orientation' in path:
                path['edge_orientations'] = path.pop('edge_orientation')
            paths.append(path)

    return paths, nodes, edges, edge_coordinates

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, edge_features=None):
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features = 5
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features = 6
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = load_new_format(file_path)
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float64)
        
        self.graph = nx.Graph()
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]

        # Add edges with index to the graph
        for (start, end), index in indexed_edges:
            self.graph.add_edge(start, end, index=index, default_orientation=(start, end))
        self.graph.add_nodes_from(self.nodes)

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        edge_idxs = torch.tensor(trajectory['edge_idxs'][:], dtype=torch.long)
        edge_orientations = torch.tensor(trajectory['edge_orientations'][:], dtype=torch.long)
        
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
        edge_idxs = nn.functional.pad(edge_idxs, (0, padding_length), value=-1)
        edge_orientations = nn.functional.pad(edge_orientations, (0, padding_length), value=0)
        
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
        
        history_one_hot_edges = nn.functional.one_hot(history_indices[valid_history_mask], num_classes=len(self.edges))
        future_one_hot_edges = nn.functional.one_hot(future_indices[valid_future_mask], num_classes=len(self.edges))
        
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
            
        # Basic History edge features = coordinates, binary encoding
        history_edge_features = history_one_hot_edges.view(-1, 1).float()
        future_edge_features = future_one_hot_edges.view(-1, 1).float()
        if 'coordinates' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
            future_edge_features = torch.cat((future_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
            self.num_edge_features = 5
        if 'edge_orientations' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, history_edge_orientations.float()), dim=1)
            future_edge_features = torch.cat((future_edge_features, future_edge_orientations.float()), dim=1)
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
            "node_in_history": node_in_history,
        }, self.graph, self.edges
        
    def __len__(self):
        return len(self.trajectories)

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
    graph = [item[1] for item in batch]
    edges = [item[2] for item in batch]
    # Extract elements for each sample and stack them, handling variable lengths
    history_indices = torch.stack([item[0]['history_indices'] for item in batch])
    future_indices = torch.stack([item[0]['future_indices'] for item in batch])
    
    history_one_hot_edges = torch.stack([item[0]['history_one_hot_edges'] for item in batch])
    future_one_hot_edges = torch.stack([item[0]['future_one_hot_edges'] for item in batch])

    # Coordinates
    history_coordinates = [item[0]['history_coordinates'] for item in batch if item[0]['history_coordinates'] is not None]
    future_coordinates = [item[0]['future_coordinates'] for item in batch if item[0]['future_coordinates'] is not None]
    
    history_edge_orientations = torch.stack([item[0]['history_edge_orientations'] for item in batch])
    future_edge_orientations = torch.stack([item[0]['future_edge_orientations'] for item in batch])
    
    history_edge_features = torch.stack([item[0]['history_edge_features'] for item in batch])
    future_edge_features = torch.stack([item[0]['future_edge_features'] for item in batch])
    
    history_one_hot_nodes = torch.stack([item[0]['node_in_history'] for item in batch])
    
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
            "history_one_hot_nodes": history_one_hot_nodes,
            "graph": graph,
            "edges": edges,
        }

import torch
import nn.functional as F
import nn as nn

class Benchmark_MLP(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, num_edges, hidden_channels, num_edge_features, max_degree):
        super(Benchmark_MLP, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.num_classes = num_classes        
        
        self.hidden_channels = hidden_channels
        self.num_layers = self.config['num_layers']
        self.max_degree = max_degree
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(self.num_edge_features, self.hidden_channels))
        for _ in range(1, self.num_layers):
            self.lin_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        
        self.output_layer = nn.Linear(self.hidden_channels, 1)  # Output one logit per neighbor

    def forward(self, history_edge_features, neighbors):
        x = history_edge_features
        for layer in self.lin_layers:
            x = F.relu(layer(x))    # (bs, num_edges, hidden_channels)
        
        logits = self.output_layer(x).squeeze(-1)  # (bs, num_edges)
    
        # Mask logits for padded neighbors
        mask = neighbors != 0
        # logits[~mask] = float('-inf')
        logits[~mask] = float(-10)
        
        preds = torch.argmax(logits, dim=-1)  # (bs,)
        return logits, preds

class EdgeRNN(nn.Module):
    def __init__(self, model_config, num_edge_features, num_edges):
        super(EdgeRNN, self).__init__()
        self.model_config = model_config
        self.input_dim = num_edge_features
        self.num_edges = num_edges
        self.hidden_dim = self.model_config['hidden_channels']
        self.num_layers = self.model_config['num_layers']
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)  # Output one logit per neighbor
    
    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)
    
    def recursive_edge_prediction(self, edge_features, initial_edge, line_graph, future_len):
        batch_size, num_edges, num_features = edge_features.size()
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(edge_features.device)
        current_edge = initial_edge
        predicted_edges = []
        logits_list = []

        for _ in range(future_len):
            neighbors = self.get_neighbors(line_graph, current_edge, edge_features.device)  # shape: [batch_size, num_edges]
            neighbor_features = self.get_neighbor_features(edge_features, neighbors)  # shape: [batch_size, num_neighbors, num_features]
            max_num_neighbors = neighbors.sum(1).max().item()
            logits, hidden = self.forward(neighbor_features, hidden)
            logits = logits.squeeze(-1)  # Remove the last dimension to match [batch_size, num_neighbors]
            
            # Mask logits
            mask = torch.arange(max_num_neighbors, device=edge_features.device).expand(len(neighbors), max_num_neighbors) < neighbors.sum(1, keepdim=True)
            logits[~mask] = float(-10)
            
            logits_list.append(logits)
            
            _, predicted_edge_idx = torch.max(logits, dim=1)
            #print("predicted_edge_idx", predicted_edge_idx)
            neighbor_indices = torch.nonzero(neighbors, as_tuple=True)
            neighbor_indices = torch.split(neighbor_indices[1], neighbors.sum(dim=1).tolist())
            max_len = max(len(s) for s in neighbor_indices)
            neighbor_indices = torch.stack([F.pad(s, (0, max_len - len(s))) for s in neighbor_indices])
            #print("Neighbor indices", neighbor_indices)
            predicted_edge = neighbor_indices.gather(1, predicted_edge_idx.unsqueeze(1)).squeeze(1)
            #print("predicted_edge", predicted_edge)

            predicted_edges.append(predicted_edge)
            current_edge = predicted_edge.unsqueeze(1)  # Update current edge for the next iteration
        return logits_list, predicted_edges
    
    def get_neighbor_features(self, edge_features, neighbors):
        batch_size, num_edges, num_features = edge_features.size()
        neighbor_list = []
        for i in range(batch_size):
            if neighbors[i].sum() == 0:
                neighbor_list.append(torch.zeros(1, num_features, dtype=torch.float))
                continue
            neighbor_indices = neighbors[i].nonzero(as_tuple=False).squeeze(1)
            neighbor_list.append(edge_features[i, neighbor_indices])
        
        neighbor_features = nn.utils.rnn.pad_sequence(neighbor_list, batch_first=True)
        
        """# Filter out the row indicating the edge in the history!
        filtered_neighbor_features = neighbor_features[~(neighbor_features[:, :, 0] == 1)]
        filtered_neighbor_features = filtered_neighbor_features.view(neighbor_features.size(0), -1, neighbor_features.size(2))
        return filtered_neighbor_features
        """
        return neighbor_features
    
    def get_neighbors(self, line_graph, edge, device):
        neighbors = []
        for e in edge:
            neighbor_set = set()
            for i in range(line_graph.edge_index.size(1)):
                if line_graph.edge_index[0, i] == e:
                    neighbor_set.add(line_graph.edge_index[1, i].item())
                elif line_graph.edge_index[1, i] == e:
                    neighbor_set.add(line_graph.edge_index[0, i].item())
            neighbor_tensor = torch.zeros(self.num_edges, dtype=torch.long, device=device)
            neighbor_tensor[list(neighbor_set)] = 1
            neighbors.append(neighbor_tensor)
        
        return torch.stack(neighbors)

encoder_model = Benchmark_MLP
encoder_model = EdgeRNN

    
data_config = {"dataset": "synthetic_2_traj",
    "train_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/tdrive_1000.h5',
    "test_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/tdrive_1001_1200.h5',
    "history_len": 5,
    "future_len": 2,
    "num_classes": 2,
    "edge_features": ['one_hot_edges', 'coordinates'],
    "one_hot_nodes": False}

model_config = {"name": "mlp_benchmark",
    "hidden_channels": 32,
    "time_embedding_dim": 16,
    "condition_dim": 16,
    "out_ch": 1,
    "num_heads": 1,
    "num_layers": 1,
    "theta": 1.0, # controls strength of conv layers in residual model
    "dropout": 0.1,
    "model_output": "logits",
    "model_prediction": "x_start",  # Options: 'x_start','xprev'
    "transition_mat_type": 'gaussian',  # Options: 'gaussian','uniform','absorbing', 'marginal_prior'
    "transition_bands": 1,
    "loss_type": "cross_entropy_x_start",  # Options: kl, cross_entropy_x_start, hybrid
    "hybrid_coeff": 0.001,  # Only used for hybrid loss type.
    "class_weights": [0.05, 0.95] # = future_len/num_edges and (num_edges - future_len)/num_edges
    }

train_config = {"batch_size": 10,
    "optimizer": "adam",
    "lr": 0.01,
    "gradient_accumulation": False,
    "gradient_accumulation_steps": 2,
    "num_epochs": 1000,
    "learning_rate_warmup_steps": 2000, # previously 10000
    "lr_decay": 0.9999, # previously 0.9999
    "log_loss_every_steps": 20,
    "save_model": False,
    "save_model_every_steps": 1000}

test_config = {"batch_size": 1, # currently only 1 works
    "model_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/synthetic_d3pm_residual_fixed/synthetic_d3pm_residual_fixed_hidden_dim_32_time_dim_16_condition_dim_16_layers_2_weights_0.1.pth',
    "number_samples": 1,
    "eval_every_steps": 1000
  }

wandb_config = {"exp_name": "tdrive_benchmark_test",
    "project": "trajectory_prediction_using_denoising_diffusion_models",
    "entity": "joeschmit99",
    "job_type": "test",
    "notes": "Benchmark test",
    "tags": ["synthetic", "MLP_Benchmark"]} 

model = Benchmark_Models(data_config, model_config, train_config, test_config, wandb_config, encoder_model)
model.train()
model_path = test_config["model_path"]
sample_list, ground_truth_hist, ground_truth_fut = model.get_samples(load_model=False, model_path=model_path, task='predict', number_samples=1)
print(sample_list)
print("\n")
print(ground_truth_hist)
print("\n")
print(ground_truth_fut)
# model.visualize_predictions(sample_list, ground_truth_hist, ground_truth_fut)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb

class AutoregressiveTrajectoryDataset(Dataset):
    def __init__(self, file_path, device):
        self.device = device
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(file_path, self.device)
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float32, device=self.device)
        
        self.num_edges = len(self.edges)
        self.graph = self.build_graph()
        #print("self graph edges", self.graph.edges(data=True))
        self.edge_list = list(self.graph.edges())
        #print("Self edge list", self.edge_list)
        self.data = []
        # Preprocess and move data to the specified device
        for trajectory in self.trajectories:
            for i in range(1, len(trajectory['edge_idxs'])):
                sequence = trajectory['edge_idxs'][:i]
                target = trajectory['edge_idxs'][i]
                
                edge_coordinates = self.edge_coordinates
                edge_coordinates_flat = torch.flatten(edge_coordinates, start_dim=1)

                # Create zero-filled tensors for sequence and target
                sequence_tensor_bin = torch.zeros(self.num_edges, device=self.device)
                target_tensor_bin = torch.zeros(self.num_edges, device=self.device)

                # Set 1 for indices in sequence and target
                sequence_tensor_bin[sequence] = 1
                feature_tensor = torch.cat((sequence_tensor_bin.unsqueeze(1), edge_coordinates_flat), dim=1)
                
                target_tensor_bin[target] = 1

                # Generate mask based on the last edge in the sequence
                mask = torch.zeros(self.num_edges, device=self.device)
                if sequence is not None:
                    last_edge_index = sequence[-1]
                    last_edge = self.edge_list[last_edge_index]
                    # Find edges that share a node with the last edge
                    neighbors = [idx for idx, edge in enumerate(self.edge_list)
                                 if (edge[0] == last_edge[0] or edge[1] == last_edge[1]
                                     or edge[0] == last_edge[1] or edge[1] == last_edge[0])]
                    mask[neighbors] = 1
                    mask[last_edge_index] = 0  # Ensure the current edge is not included in the mask

                # self.data.append((sequence_tensor, target_tensor, mask))
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
                target_tensor = torch.tensor(target, dtype=torch.long)
                self.data.append((sequence_tensor_bin, target_tensor_bin, mask, sequence_tensor, target_tensor))
                #self.data.append((feature_tensor, target_tensor_bin, mask, sequence_tensor, target_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Directly return the preprocessed data, which is already on the correct device
        return self.data[idx]
    
    @staticmethod
    def load_new_format(file_path, device):
        paths = []
        with h5py.File(file_path, 'r') as new_hf:
            node_coordinates = torch.tensor(new_hf['graph']['node_coordinates'][:], dtype=torch.float, device=device)
            #edges = torch.tensor(new_hf['graph']['edges'][:], dtype=torch.long, device=device)
            edges = new_hf['graph']['edges'][:]
            edge_coordinates = node_coordinates[edges]
            nodes = [(i, {'pos': torch.tensor(pos, device=device)}) for i, pos in enumerate(node_coordinates)]
            #edges = [(torch.tensor(edge[0], device=device), torch.tensor(edge[1], device=device)) for edge in edges]
            edges = [tuple(edge) for edge in edges]

            for i in tqdm(new_hf['trajectories'].keys()):
                path_group = new_hf['trajectories'][i]
                path = {attr: torch.tensor(path_group[attr][()], device=device) for attr in path_group.keys() if attr in ['coordinates', 'edge_idxs', 'edge_orientations']}
                paths.append(path)
            
        return paths, nodes, edges, edge_coordinates
    
    def build_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in indexed_edges:
            graph.add_edge(start, end, index=index, default_orientation=(start, end))
        return graph
    
    
class AlternativeAutoregressiveTrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, device):
        self.device = device
        self.history_len = history_len
        self.future_len = future_len
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(file_path, self.device)
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float32, device=self.device)
        
        self.num_edges = len(self.edges)
        self.graph = self.build_graph()
        self.edge_list = list(self.graph.edges())
        self.data = []
        # Preprocess and move data to the specified device
        for trajectory in self.trajectories:
            for i in range(1, self.history_len):
                sequence = trajectory['edge_idxs'][:i]
                target = trajectory['edge_idxs'][i]
                
                edge_coordinates = self.edge_coordinates[sequence]
                edge_coordinates_flat = torch.flatten(edge_coordinates, start_dim=1)
                
                feature_tensor = torch.cat((sequence.unsqueeze(1), edge_coordinates_flat), dim=1)

                # Create zero-filled tensors for target
                target_tensor_bin = torch.zeros(self.num_edges, device=self.device)
                target_tensor_bin[target] = 1

                # Generate mask based on the last edge in the sequence
                mask = torch.zeros(self.num_edges, device=self.device)
                if sequence is not None:
                    last_edge_index = sequence[-1]
                    last_edge = self.edge_list[last_edge_index]
                    # Find edges that share a node with the last edge
                    neighbors = [idx for idx, edge in enumerate(self.edge_list)
                                 if (edge[0] == last_edge[0] or edge[1] == last_edge[1]
                                     or edge[0] == last_edge[1] or edge[1] == last_edge[0])]
                    mask[neighbors] = 1
                    mask[last_edge_index] = 0  # Ensure the current edge is not included in the mask

                # self.data.append((sequence_tensor, target_tensor, mask))
                sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
                target_tensor = torch.tensor(target, dtype=torch.long)
                self.data.append((feature_tensor, target_tensor, target_tensor_bin, mask))
                # self.data.append((sequence_tensor_bin, target_tensor_bin, mask, sequence_tensor, target_tensor))
                #self.data.append((feature_tensor, target_tensor_bin, mask, sequence_tensor, target_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Directly return the preprocessed data, which is already on the correct device
        return self.data[idx]
    
    @staticmethod
    def load_new_format(file_path, device):
        paths = []
        with h5py.File(file_path, 'r') as new_hf:
            node_coordinates = torch.tensor(new_hf['graph']['node_coordinates'][:], dtype=torch.float, device=device)
            #edges = torch.tensor(new_hf['graph']['edges'][:], dtype=torch.long, device=device)
            edges = new_hf['graph']['edges'][:]
            edge_coordinates = node_coordinates[edges]
            nodes = [(i, {'pos': torch.tensor(pos, device=device)}) for i, pos in enumerate(node_coordinates)]
            #edges = [(torch.tensor(edge[0], device=device), torch.tensor(edge[1], device=device)) for edge in edges]
            edges = [tuple(edge) for edge in edges]

            for i in tqdm(new_hf['trajectories'].keys()):
                path_group = new_hf['trajectories'][i]
                path = {attr: torch.tensor(path_group[attr][()], device=device) for attr in path_group.keys() if attr in ['coordinates', 'edge_idxs', 'edge_orientations']}
                paths.append(path)
            
        return paths, nodes, edges, edge_coordinates
    
    def build_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in indexed_edges:
            graph.add_edge(start, end, index=index, default_orientation=(start, end))
        degrees = graph.degree()
        average_degree = sum(deg for _, deg in degrees) / float(len(graph))
        self.avg_degree = average_degree
        return graph
    
def collate_fn(batch):
    # 'batch' is a list of tuples (sequence_tensor, target_tensor)
    feature_tensor, targets, target_bin, masks = zip(*batch)
    
    # Pad sequences
    padded_feature_tensor = pad_sequence(feature_tensor, batch_first=True, padding_value=0)
    
    # Stack targets - assuming targets are already all the same length or are scalars
    targets = torch.stack(targets)
    
    target_bin = torch.stack(target_bin)
    
    masks = torch.stack(masks)
    
    lengths = torch.tensor([len(seq) for seq in padded_feature_tensor])
    
    return padded_feature_tensor, targets, target_bin, masks, lengths

class EdgeModel(nn.Module):
    def __init__(self, hidden_size, num_features, num_layers, num_edges):
        super().__init__()
        hidden_size = 16
        self.rnn = nn.RNN(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_edges)

    def forward(self, x, hidden, mask, lengths):
        #print("Sequence x", x)
        #print(x.shape)
        # x should be of shape (seq_len, input_size)
        # print("Sequence", x)
        # x = x.unsqueeze(-1)    # Add a dimension for the input size
        # x has shape [bs, seq_len, num_features]
        #x, hidden = self.rnn(x, hidden)    # for unbatched data: x: (seq_len, hidden_size=num_edges), hidden: (num_layers, hidden_size=num_edges), else x: (batch_size, seq_len, hidden_size), hidden: (batch_size, seq_len, hidden_size)
        # print("out shape", x.shape)
        #print("hidden shape", hidden.shape)
        # x of shape (num_edges, num_edges)
        #print("x", x.shape)
        #print("hidden", hidden.shape)
        #logits = self.fc(x)
        #print("Logits shape", logits.shape)
        
        # x has shape [bs, seq_len, num_features]
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)  # hidden has size [num_layers, batch_size, hidden_size]
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        hidden = hidden[-1]  # Take the last hidden state
        logits = self.fc(hidden)    # logits have size [batch_size, num_edges]
        
        probabilities = self.apply_mask(logits, mask)
        preds = torch.argmax(torch.softmax(probabilities, dim=-1), dim=-1)
        return probabilities, preds

    def apply_mask(self, logits, mask):
        '''masked_logits = logits.masked_fill(mask == 0, float('-inf'))  # Set logits for non-neighbors to -inf
        probabilities = torch.softmax(masked_logits, dim=-1)
        return probabilities'''
        masked_logits = logits.masked_fill(mask == 0, float(-10))
        return masked_logits


class Train_Model(nn.Module):
    def __init__(self, wandb_config, hidden_size, num_features, model, train_file_path, val_file_path):
        super(Train_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.hidden_size = hidden_size
        self.num_features = num_features
        
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.num_epochs = 100
        self.lr = 0.005
        self.learning_rate_warmup_steps = 1000
        self.num_layers = 2
        
        # WandB
        self.wandb_config = wandb_config
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.wandb_config['project'],
            entity=self.wandb_config['entity'],
            notes=self.wandb_config['notes'],
            job_type=self.wandb_config['job_type'],
            # config={**self.data_config, **self.diffusion_config, **self.model_config, **self.train_config}
        )
        self.exp_name = self.wandb_config['exp_name']
        wandb.run.name = self.wandb_config['run_name']

        self.model = model
        self._build_train_dataloader()
        self._build_val_dataloader()
        self._build_model()
        self._build_optimizer()

    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            current_lr = self.scheduler.get_last_lr()[0]
            wandb.log({"Epoch": epoch, "Learning Rate": current_lr})
            self.optimizer.zero_grad()
            total_loss = 0
            acc = 0
            for features, target, target_bin, mask, lengths in self.train_dataloader:
                # Sequence, target, mask = (bs=1, num_edges)
                #sequence_bin = sequence_bin.squeeze(0)
                batch_size = features.size(0)
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
                
                probabilities, preds = self.model(features, hidden, mask, lengths)
                loss = F.cross_entropy(probabilities, target)
                total_loss += loss.item()
                acc += (preds == target).sum().item() / batch_size
                loss.backward()
                self.optimizer.step()
            print("Avg Loss", round(total_loss / len(self.train_dataloader), 5))
            wandb.log({"Epoch": epoch, "Average Train Loss": total_loss / len(self.train_dataloader)})
            wandb.log({"Epoch": epoch, "Train Accuracy": acc / len(self.train_dataloader)})
            wandb.log({"Epoch": epoch, "Random Baseline": 1 / self.train_dataset.avg_degree})

    def eval(self):
        sequences = []
        targets = []
        preds = []
        acc = 0
        for features, target, target_bin, mask, lengths in self.val_dataloader:
            batch_size = features.size(0)
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
            probabilities, pred = self.model(features, hidden, mask, lengths)
            sequence = [torch.tensor([row[i, 0] for i in range(row.size(0)) if row[i, 0] != 0]) for row in features]
            sequences.append(sequence)
            targets.append(target)
            preds.append(pred)
            acc += (pred == target).sum().item() / batch_size
        avg_acc = acc / len(self.val_dataloader)
        wandb.log({"Val Accuracy": avg_acc})
        print("Val Accuracy:", avg_acc)
        return {"Sequences": sequences, "Targets": targets, "Predictions": preds}
    
    def _build_train_dataloader(self):
        # train_dataset = AutoregressiveTrajectoryDataset(self.train_file_path, device=self.device)
        self.train_dataset = AlternativeAutoregressiveTrajectoryDataset(self.train_file_path, history_len=5, future_len=2, device=self.device)
        self.num_edges = self.train_dataset.num_edges
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
    def _build_val_dataloader(self):
        # self.val_dataset = AutoregressiveTrajectoryDataset(self.val_file_path, device=self.device)
        self.val_dataset = AlternativeAutoregressiveTrajectoryDataset(self.val_file_path, history_len=5, future_len=2, device=self.device)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    def _build_model(self):
        self.model = self.model(self.hidden_size, self.num_features, self.num_layers, self.num_edges)
        self.model.to(self.device)
        
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


# Example usage
train_file_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/tdrive_1000.h5'
val_file_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/tdrive_1001_1200.h5'

wandb_config = {"exp_name": "benchmark_test",
                "run_name": "Coords, L=History_size",
                "project": "trajectory_prediction_using_denoising_diffusion_models",
                "entity": "joeschmit99",
                "job_type": "test",
                "notes": "",
                "tags": ["synthetic", "benchmark_rnn"]} 

model = Train_Model(wandb_config, hidden_size=16, num_features=5, model=EdgeModel, train_file_path=train_file_path, val_file_path=val_file_path)
model.train()
eval_dict = model.eval()
print("History sequence", eval_dict['Sequences'])
print("Target", eval_dict['Targets'])
print("Pred", eval_dict['Predictions'])
