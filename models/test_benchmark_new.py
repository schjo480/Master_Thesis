import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import networkx as nx
import numpy as np
torch.set_printoptions(threshold=10_000)


class TrajectoryDataset(Dataset):
    def __init__(self, file_path, edge_features, device, history_len, future_len, true_future, mode='train'):
        self.edge_features = edge_features
        self.device = device
        self.mode = mode
        self.history_len = history_len
        self.future_len = future_len
        if 'road_type' in self.edge_features:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates, self.road_type = self.load_new_format(file_path, self.edge_features, self.device)
            self.road_type = torch.tensor(self.road_type, dtype=torch.float64, device=self.device)
            self.num_road_types = self.road_type.size(1)
        else:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(file_path, self.edge_features, self.device)

        self.edge_coordinates = torch.cat((self.edge_coordinates, torch.zeros((2, self.edge_coordinates.size(1), 2), device=self.device)))  # Padding for stop token and padding token
        self.num_edges = len(self.edges)
        self.stop_token = len(self.edges)
        self.padding_value = len(self.edges) + 1
        self.adjacency_matrix = self.creat_adjaency_matrix()
        self.true_future = true_future

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

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        if self.mode == 'train':
            edge_idxs = self.trajectories[idx]['edge_idxs']
            # Append stop token if the sequence is shorter than the desired length
            input_seq = torch.tensor(edge_idxs, dtype=torch.long, device=self.device)
            target_seq = torch.cat((edge_idxs[1:], torch.tensor([self.stop_token], device=self.device)))
            
            masks = [self.adjacency_matrix[e] for e in input_seq]
            masks = torch.stack(masks)
            feature_tensor = [self.generate_edge_features(input_seq[:i]) for i in range(1, len(input_seq)+1)]
            feature_tensor = torch.stack(feature_tensor)
        
        else:
            edge_idxs = self.trajectories[idx]['edge_idxs']
            true_future_len = len(edge_idxs) - self.history_len

            input_seq = torch.tensor(edge_idxs[:self.history_len], dtype=torch.long, device=self.device)
            # For validation/test, use true future length
            if self.true_future:
                target_seq = torch.tensor(edge_idxs[1:], dtype=torch.long, device=self.device)
            else:
                target_seq = torch.cat((edge_idxs[1:self.history_len+self.future_len], torch.tensor([self.stop_token], device=self.device)))
            
            masks = [self.adjacency_matrix[e] for e in input_seq]
            masks = torch.stack(masks)
            
            feature_tensor = [self.generate_edge_features(input_seq[:i]) for i in range(1, len(input_seq)+1)]
            feature_tensor = torch.stack(feature_tensor)
        
        return input_seq, target_seq, self.padding_value, masks, feature_tensor

    def creat_adjaency_matrix(self):
        edge_to_nodes = {}
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in indexed_edges:
            G.add_edge(start, end, index=index, default_orientation=(start, end))
            edge_to_nodes.update({index: set([start, end])})

        degrees = G.degree()
        average_degree = sum(deg for _, deg in degrees) / float(len(G))
        self.avg_degree = average_degree
        adjacency_matrix = torch.zeros((self.stop_token+2, self.stop_token+2), dtype=torch.bool)
    
        # Populate the adjacency matrix
        for edge1, nodes1 in edge_to_nodes.items():
            for edge2, nodes2 in edge_to_nodes.items():
                if edge1 != edge2 and not nodes1.isdisjoint(nodes2):
                    adjacency_matrix[edge1, edge2] = True
                    adjacency_matrix[edge2, edge1] = True  # Ensure symmetry since the graph is undirected

        stop_token_index = self.stop_token
        #adjacency_matrix[-1, :] = True
        adjacency_matrix[:, -2] = True  # Stop token must be reachable from any edge
        adjacency_matrix[:, -1] = True  # Padding token must be reachable from any edge
        
        return adjacency_matrix.to(self.device)

    def convert_edge_indices_to_node_paths(self, path_edges, last_history_edge):
        """
        Converts paths of edge indices to paths of node indices, using the last_history_edge to determine the correct
        starting node.

        :param path_edges: List of paths where each path is a list of edge indices
        :param last_history_edge: The last edge before path_edges begin, used to determine the starting node
        :return: List of paths with node indices
        """

        node_path = []

        # The first edge in the path
        first_edge = self.edges[path_edges[0]]
        last_history_edge = self.edges[int(last_history_edge.item())]
        # Determine the starting node using the last_history_edge
        # last_history_edge is a tuple (node_a, node_b)
        if last_history_edge[1] == first_edge[0]:
            start_node = first_edge[0]
        elif last_history_edge[1] == first_edge[1]:
            start_node = first_edge[1]
        elif last_history_edge[0] == first_edge[0]:
            start_node = first_edge[0]
        elif last_history_edge[0] == first_edge[1]:
            start_node = first_edge[1]
        else:
            # If no direct connection is found, fall back to the first node of the first edge
            start_node = first_edge[0]

        # Initialize the node path with the determined starting node
        node_path.append(start_node)

        # Initialize the last node added to the path
        last_node = start_node

        # Process each edge in the path
        for edge_idx in path_edges:
            edge = self.edges[edge_idx]
            if edge[0] == last_node:
                # If the first node of the current edge matches the last added node
                next_node = edge[1]
            else:
                # If the second node of the current edge matches the last added node
                next_node = edge[0]

            node_path.append(next_node)
            last_node = next_node  # Update the last node added to the path


        return node_path
    
    def generate_edge_features(self, history_indices):
        # Binary on/off edges
        # ensure indices are integers
        history_indices = history_indices.long()
        valid_history_mask = history_indices >= 0
        history_one_hot_edges = torch.nn.functional.one_hot(history_indices[valid_history_mask], num_classes=self.padding_value+1)
        
        # Sum across the time dimension to count occurrences of each edge
        history_one_hot_edges = history_one_hot_edges.sum(dim=0)  # (num_edges,)
        
        # Basic History edge features = coordinates, binary encoding
        history_edge_features = history_one_hot_edges.view(-1, 1).float()
        
        if 'coordinates' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
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
            if torch.isnan(cosines).any():
                cosines = torch.nan_to_num(cosines, nan=0.0)
            history_edge_features = torch.cat((history_edge_features, cosines.float()), dim=1)

        history_edge_features = torch.nan_to_num(history_edge_features, nan=0.0)
        if 'one_hot_edges' not in self.edge_features:
            history_edge_features = history_edge_features[:, 1:]
        
        return history_edge_features
    
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
        if len(trajectory_edges) == 1:
            return trajectory_edges[0, 0], trajectory_edges[0, 1]
        
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

def collate_fn(batch):
    inputs, targets, padding_value, masks_list, feature_tensor_list = zip(*batch)
    # Pad sequences using the defined padding value which does not overlap with valid edge indices or the stop token
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value[0])
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=padding_value[0])
    
    max_length = max(len(masks) for masks in masks_list)
    padded_masks = torch.stack([torch.cat([mask, torch.zeros(max_length - mask.size(0), mask.size(1), dtype=torch.bool, device=mask.device)]) for mask in masks_list])  # (batch_size, seq_length, num_edges)
    padded_feature_tensors = torch.stack([torch.cat([feature_tensor, torch.zeros(max_length - feature_tensor.size(0), feature_tensor.size(1), feature_tensor.size(2), dtype=torch.float, device=feature_tensor.device)]) for feature_tensor in feature_tensor_list])    # (batch_size, seq_length, num_edges, num_edge_features)
    
    return padded_inputs, padded_targets, padded_masks, padded_feature_tensors

# TODO: Sample for true future length
# TODO: Include edge features
# --> Adjust: one edge_feature tensor per subsequence, similar to masking opration!!


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EdgeRNN(nn.Module):
    def __init__(self, num_edge_features_rnn, num_edge_features, hidden_size, output_size, num_layers, dropout, model_type='rnn'):
        super(EdgeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.feature_encoding_1 = nn.Linear(num_edge_features, self.hidden_size)
        self.feature_encoding_2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden, masks, feature_tensor=None):
        feature_encoding = F.relu(self.feature_encoding_1(feature_tensor))  # (batch_size, seq_length, num_edges, hidden_size)
        feature_encoding = self.feature_encoding_2(feature_encoding).squeeze(-1)   # (batch_size, seq_length, num_edges)
        #print("Input seq", x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # (batch_size, seq_length, num_edges)
        out = out + feature_encoding
        #print("Masks", torch.argwhere(masks == 1))
        out = out.masked_fill(~masks, float(-10000))
        # Apply softmax to get probabilities
        out = F.log_softmax(out, dim=-1)

        return out, hidden


    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        if self.model_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import networkx as nx
import numpy as np
import wandb
import os
import logging
import time

class Trainer:
    def __init__(self, wandb_config, data_config, model, model_type, dataloader, val_dataloader, padding_value, stop_token, num_edge_features, future_len, history_len, baseline_acc, device, true_future, learning_rate=0.005):
        self.device = device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.nodes = dataloader.dataset.nodes
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.padding_value = padding_value
        self.stop_token = stop_token
        self.baseline_acc = baseline_acc
        self.num_edge_features = num_edge_features
        self.future_len = future_len
        self.history_len = history_len
        self.model_type = model_type
        self.true_future = true_future
        
        self.data_config = data_config
        self.wandb_config = wandb_config
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.wandb_config['project'],
            entity=self.wandb_config['entity'],
            notes=self.wandb_config['notes'],
            job_type=self.wandb_config['job_type'],
            config={}
        )
        self.exp_name = self.wandb_config['exp_name']
        
        run_name = f'{self.exp_name}_{self.model_type}_hist{self.history_len}_fut{self.future_len}'
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

    def train(self, epochs):
        self.model.train()
        for epoch in tqdm(range(epochs)):
            total_loss = 0
            total_val_loss = 0
            total_correct = 0
            total_elements = 0
            b_ct = 1
            for inputs, targets, masks, feature_tensor in self.dataloader:
                inputs, targets, masks, feature_tensor = inputs.to(self.device), targets.to(self.device), masks.to(self.device), feature_tensor.to(self.device)
                batch_size = inputs.size(0)
                seq_length = inputs.size(1)
                inputs = inputs.unsqueeze(-1).float().to(self.device)   # (batch_size, seq_length, 1)
                self.optimizer.zero_grad()

                hidden = self.model.init_hidden(batch_size, self.device)
                if self.num_edge_features > 0:
                    output, hidden = self.model(inputs, hidden, masks, feature_tensor)
                else:
                    output, hidden = self.model(inputs, hidden, masks)

                print("Output", output.shape)
                # Reshape output and targets to fit CE loss requirements
                output = output.view(-1, output.size(-1))  # [batch_size * seq_length, output_size]
                targets = targets.view(-1)  # [batch_size * seq_length]
                
                # Calculate loss and accuracy
                loss = self.loss_function(output, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                # Calculate accuracy
                with torch.no_grad():
                    predictions = output.argmax(dim=-1)
                    mask = targets != self.padding_value
                    total_correct += (predictions[mask] == targets[mask]).sum().item()
                    total_elements += mask.sum().item()
                
                b_ct += 1

            avg_loss = total_loss / len(self.dataloader)
            accuracy = total_correct / total_elements if total_elements > 0 else 0
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            wandb.log({"Epoch": epoch, "Average Train Loss": avg_loss})
            wandb.log({"Epoch": epoch, "Train Accuracy": accuracy})
            
            if epoch % 10 == 0:
                for inputs, targets, masks, feature_tensor in self.val_dataloader:
                    inputs, targets, masks, feature_tensor = inputs.to(self.device), targets.to(self.device), masks.to(self.device), feature_tensor.to(self.device)
                    batch_size = inputs.size(0)
                    inputs = inputs.unsqueeze(-1).float().to(self.device)   # (batch_size, seq_length, 1)

                    hidden = self.model.init_hidden(batch_size, self.device)
                    if self.num_edge_features > 0:
                        output, hidden = self.model(inputs, hidden, masks, feature_tensor)
                    else:
                        output, hidden = self.model(inputs, hidden, masks)

                    # Reshape output and targets to fit CE loss requirements
                    output = output.view(-1, output.size(-1))  # [batch_size * seq_length, output_size]
                    targets = targets.view(-1)  # [batch_size * seq_length]
                    
                    # Calculate loss and accuracy
                    loss = self.loss_function(output, targets)
                    total_val_loss += loss.item()
                print(f'Epoch {epoch+1}, Average Validation Loss: {total_val_loss / len(self.val_dataloader):.4f}')
                wandb.log({"Epoch": epoch, "Average Validation Loss": total_val_loss / len(self.val_dataloader)})

        self.save_model()
        
    def test(self, test_dataloader, mode='fixed', max_prediction_length=None):
        """
        mode: 'fixed' or 'dynamic'
            - 'fixed': predict the next x edges
            - 'dynamic': predict until stop token or padding token is predicted
        max_prediction_length: maximum number of edges to predict (used in 'fixed' mode)
        """
        self.model.eval()

        total_distance = 0  # For ADE
        total_ade_steps = 0
        total_final_distance = 0  # For FDE
        total_fde_sequences = 0
        total_true_length = 0
        total_length = 0
        ct = 0
        gt_hist = []
        gt_fut = []
        pred_list = []

        with torch.no_grad():
            for inputs, targets, masks, feature_tensor in test_dataloader:
                batch_size = inputs.size(0)
                # Initialize hidden state
                hidden = self.model.init_hidden(batch_size, self.device)
                inputs = inputs.unsqueeze(-1).float()  # (batch_size, seq_length, 1), only necessary if no features provided
                current_inputs = inputs.clone()
                if self.model.model_type == 'lstm':
                    h = (hidden[0].clone(), hidden[1].clone())
                else:
                    h = hidden.clone()
                current_masks = masks.clone()
                current_feature_tensor = feature_tensor.clone()
                pred_seq = []
                target_seq = []
                last_edge = []
                step = 0
                edge_acc = 0
                # Necessary for breaking when stop token is predicted
                while True:
                    # Forward pass
                    true_future_lengths = [len(targets[i]) for i in range(batch_size)]
                    last_edge.append(current_inputs[:, -1].reshape(batch_size, 1))
                    if self.num_edge_features > 0:
                        out, h = self.model(current_inputs, h, current_masks, current_feature_tensor)
                    else:
                        out, h = self.model(current_inputs, h, current_masks)
                    out = out.squeeze(1)  # Shape: [1, num_edges + 2]
                    next_edge = out.argmax(dim=-1)  # Predicted edge index
                    next_edge = next_edge[:, -1]
                    
                    # Update targets
                    if targets.size(1) > self.history_len+step-1:
                        current_targets = targets[:, self.history_len+step-1]
                    else:
                        current_targets = torch.tensor([self.stop_token] * batch_size, device=self.device)

                    if len(target_seq) == 0:
                        target_seq = current_targets.reshape(batch_size, 1)
                    else:
                        target_seq = torch.cat((target_seq, current_targets.reshape(batch_size, 1)), dim=1)
                    edge_acc += (next_edge == current_targets).sum().item() / (batch_size)
                    next_edge = next_edge.reshape(batch_size, 1)    # (batch_size, 1)
                    if len(pred_seq) == 0:
                        pred_seq = next_edge
                    else:
                        # Concatenate new predictions horizontally along axis 1
                        pred_seq = torch.cat((pred_seq, next_edge), dim=1)  # Shape: [batch_size, step]

                    if torch.all((next_edge == self.padding_value) | (next_edge == self.stop_token)):
                        break
                    # Update inputs
                    current_inputs = torch.cat([current_inputs, next_edge.reshape(batch_size, 1, 1)], dim=1)
                    
                    # Update masks
                    new_masks = [self.dataloader.dataset.adjacency_matrix[e] for e in next_edge]
                    new_masks = torch.stack(new_masks).reshape(batch_size, 1, -1)
                    current_masks = torch.cat((current_masks, new_masks), dim=1)
                    
                    # Update feature tensor
                    new_feature_tensor = [self.dataloader.dataset.generate_edge_features(current_inputs.squeeze(-1)[i]) for i in range(batch_size)]
                    new_feature_tensor = torch.stack(new_feature_tensor).unsqueeze(1)
                    current_feature_tensor = torch.cat((current_feature_tensor, new_feature_tensor), dim=1)
                    step += 1
                    # Check for maximum prediction length
                    if self.true_future:
                        if step >= max(true_future_lengths):
                            break
                    else:
                        if mode == 'fixed' and step >= max_prediction_length:
                            break
                    
                # Calculate ade and fde
                for i in range(batch_size):
                    last_hist_edge = last_edge[0][i]
                    if self.true_future:
                        pred_seq_i = pred_seq[i][:true_future_lengths[i]]
                    else:
                        pred_seq_i = pred_seq[i]
                    pred_seq_i = pred_seq_i[pred_seq_i != self.padding_value]
                    pred_seq_i = pred_seq_i[pred_seq_i != self.stop_token]
                    targets_i = target_seq[i]
                    targets_i = targets_i[targets_i != self.padding_value]
                    targets_i = targets_i[targets_i != self.stop_token]
                    gt_hist.append(inputs[i].cpu().numpy())
                    gt_fut.append(targets_i.cpu().numpy())
                    pred_list.append(pred_seq_i.cpu().numpy())
                    total_true_length += len(targets_i)
                    total_length += len(pred_seq_i)
                    ct += 1
                    #print("Pred seq:", pred_seq_i)
                    #print("Targets:", targets_i)
                    if int(last_hist_edge.item()) >= self.dataloader.dataset.stop_token:
                        continue
                    if len(pred_seq_i) == 0 and len(targets_i) == 0:
                        continue
                    if len(pred_seq_i) == 0:
                        target_nodes = self.dataloader.dataset.convert_edge_indices_to_node_paths(targets_i, last_hist_edge)
                        for i in range(len(target_nodes)):
                            total_distance += torch.norm(self.nodes[target_nodes[i]][1]['pos'] - self.nodes[target_nodes[0]][1]['pos'])
                            total_ade_steps += 1
                        total_final_distance += torch.norm(self.nodes[target_nodes[-1]][1]['pos'] - self.nodes[target_nodes[0]][1]['pos'])
                        total_fde_sequences += 1
                        continue
                    if len(targets_i) == 0:
                        pred_nodes = self.dataloader.dataset.convert_edge_indices_to_node_paths(pred_seq_i, last_hist_edge)
                        for i in range(len(pred_nodes)):
                            total_distance += torch.norm(self.nodes[pred_nodes[i]][1]['pos'] - self.nodes[pred_nodes[0]][1]['pos'])
                            total_ade_steps += 1
                        total_final_distance += torch.norm(self.nodes[pred_nodes[-1]][1]['pos'] - self.nodes[pred_nodes[0]][1]['pos'])
                        total_fde_sequences += 1
                        continue
                    if len(pred_seq_i) > 0 and len(targets_i) > 0:
                        pred_nodes = self.dataloader.dataset.convert_edge_indices_to_node_paths(pred_seq_i, last_hist_edge)
                        target_nodes = self.dataloader.dataset.convert_edge_indices_to_node_paths(targets_i, last_hist_edge)
                        max_len = max(len(pred_nodes), len(target_nodes))
                        if len(pred_nodes) < len(target_nodes):
                            for i in range(len(target_nodes)):
                                if i < len(pred_nodes):
                                    total_distance += torch.norm(self.nodes[pred_nodes[i]][1]['pos'] - self.nodes[target_nodes[i]][1]['pos'])
                                else:
                                    total_distance += torch.norm(self.nodes[target_nodes[i]][1]['pos'] - self.nodes[pred_nodes[-1]][1]['pos'])
                        elif len(pred_nodes) > len(target_nodes):
                            for i in range(len(pred_nodes)):
                                if i < len(target_nodes):
                                    total_distance += torch.norm(self.nodes[pred_nodes[i]][1]['pos'] - self.nodes[target_nodes[i]][1]['pos'])
                                else:
                                    total_distance += torch.norm(self.nodes[pred_nodes[i]][1]['pos'] - self.nodes[target_nodes[-1]][1]['pos'])
                        else:
                            for i in range(len(pred_nodes)):
                                total_distance += torch.norm(self.nodes[pred_nodes[i]][1]['pos'] - self.nodes[target_nodes[i]][1]['pos'])

                        total_ade_steps += max_len
                        total_final_distance += torch.norm(self.nodes[pred_nodes[-1]][1]['pos'] - self.nodes[target_nodes[-1]][1]['pos'])
                        total_fde_sequences += 1
                        continue
                
            ade = total_distance / total_ade_steps if total_ade_steps > 0 else 0
            fde = total_final_distance / total_fde_sequences if total_fde_sequences > 0 else 0
            wandb.log({"ADE": ade})
            wandb.log({"FDE": fde})
            print(f'ADE: {ade:.4f}')
            print(f'FDE: {fde:.4f}')
            print("Average prediction length:", total_length / ct)
            print("Average true length:", total_true_length / ct)
            path = os.path.join(self.model_dir, 
                                 f'{self.exp_name}_{self.model_type}_hist{self.history_len}_fut{self.future_len}_')
            torch.save(gt_hist, path + 'gt_hist.pth')
            torch.save(gt_fut, path + 'gt_fut.pth')
            torch.save(pred_list, path + 'pred_list.pth')
            
    def save_model(self):
        model_path = os.path.join(self.model_dir, 
                                 f'{self.exp_name}_{self.model_type}_hist{self.history_len}_fut{self.future_len}.pth')
        torch.save(self.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        wandb.save(model_path)
        self.log.info(f'Model saved to {model_path}')
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f'Model loaded from {model_path}')
        self.log.info(f'Model loaded from {model_path}')
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

    
data_config = {"dataset": "synthetic_20_traj",
    "train_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_20_traj.h5',
    "val_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_20_traj.h5',
    "test_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_20_traj.h5',
    "history_len": 5,
    "future_len": 2,
    "num_edge_features": 8,
    "edge_features": ['one_hot_edges', 'coordinates', 'pw_distance', 'edge_length', 'edge_angles']
    }

model_config = {"model_type": "rnn",
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.1,
    }

training_config = {"batch_size": 2,
    "optimizer": "adam",
    "lr": 0.002,
    "num_epochs": 20}

testing_config = {"batch_size": 2,
                  "true_future": False}

wandb_config = {"exp_name": "test_benchmark",
                "run_name": "test_benchmark",
    "project": "trajectory_prediction_using_denoising_diffusion_models",
    "entity": "joeschmit99",
    "job_type": "test",
    "notes": "",
    "tags": ["synthetic", "rnn"]} 
    

train_dataset = TrajectoryDataset(file_path=data_config['train_data_path'],
                                      edge_features=data_config['edge_features'],
                                      device=device,
                                      history_len=data_config['history_len'],
                                      future_len=data_config['future_len'],
                                      true_future=testing_config['true_future'],
                                      mode='train')
train_dataloader = DataLoader(train_dataset,
                            batch_size=training_config['batch_size'],
                            shuffle=True, collate_fn=collate_fn)

val_dataset = TrajectoryDataset(file_path=data_config['val_data_path'],
                            edge_features=data_config['edge_features'],
                            device=device,
                            history_len=data_config['history_len'],
                            future_len=data_config['future_len'],
                            true_future=testing_config['true_future'],
                            mode='train')
val_dataloader = DataLoader(val_dataset,
                        batch_size=testing_config['batch_size'],
                        shuffle=True, 
                        collate_fn=collate_fn)

output_size = train_dataset.stop_token + 2
rnn_model = EdgeRNN(num_edge_features_rnn=1, 
                num_edge_features=data_config['num_edge_features'], 
                hidden_size=model_config['hidden_size'], 
                output_size=output_size, 
                num_layers=model_config['num_layers'], 
                dropout=model_config['dropout'], 
                model_type=model_config['model_type'])

model = Trainer(wandb_config,
            data_config,
            rnn_model,
            model_type=model_config['model_type'],
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            padding_value=train_dataset.padding_value,
            stop_token=train_dataset.stop_token,
            num_edge_features=data_config['num_edge_features'], 
            future_len=data_config['future_len'], history_len=data_config['history_len'],
            baseline_acc=1 / train_dataset.avg_degree, 
            device=device, 
            true_future=testing_config['true_future'],
            learning_rate=training_config['lr'])

model.train(epochs=training_config['num_epochs'])
#model.load_model('/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/experiments/pneuma_rnn/pneuma_rnn_gru_hist5_fut10.pth')

test_dataset = TrajectoryDataset(file_path=data_config['test_data_path'],
                            edge_features=data_config['edge_features'],
                            device=device,
                            history_len=data_config['history_len'],
                            future_len=data_config['future_len'],
                            true_future=testing_config['true_future'],
                            mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=testing_config['batch_size'], shuffle=True, collate_fn=collate_fn)
model.test(test_dataloader, mode='fixed', max_prediction_length=data_config['future_len'])
print("Stop token:", train_dataset.stop_token)
print("Padding value:", train_dataset.padding_value)
