import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import h5py
from tqdm import tqdm
import networkx as nx
import numpy as np

class TrajectoryDataset(Dataset):
    """
    Dataset class for trajectory prediction using RNNs.
    """
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
        self.adjacency_matrix = self.create_adjaency_matrix()
        self.true_future = true_future

    @staticmethod
    def load_new_format(file_path, device):
        """
        Load trajectory data in a new format from a file.
        Args:
            file_path (str): The path to the .h5 file containing the trajectory data.
            device (torch.device): The device to load the data onto.
        Returns:
            tuple: A tuple containing the following elements:
                - paths (list): A list of dictionaries representing the paths.
                - nodes (list): A list of tuples representing the nodes.
                - edges (list): A list of tuples representing the edges.
                - edge_coordinates (torch.Tensor): A tensor representing the edge coordinates.
        """
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
            return paths, nodes, edges, edge_coordinates
    
    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        """
        Retrieves the item at the given index from the dataset.
        Parameters:
            idx (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing the input sequence, target sequence, padding value, masks, and feature tensor.
                - input_seq (torch.Tensor): The input edge sequence.
                - target_seq (torch.Tensor): The target edge sequence.
                - padding_value: The padding value.
                - masks (torch.Tensor): The masks for each sequence.
                - feature_tensor (torch.Tensor): The feature tensor.
        """
        
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

    def create_adjaency_matrix(self):
        """
        Returns:
            torch.Tensor: The adjacency matrix for the edges in the graph.
        """
        
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
