import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import networkx as nx
import numpy as np

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, nodes, edges, future_len, edge_features=None):
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.trajectories = h5py.File(file_path, 'r')
        self.keys = list(self.trajectories.keys())
        
        self.nodes = nodes
        self.edges = edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

    def __getitem__(self, idx):
        trajectory_name = self.keys[idx]
        trajectory = self.trajectories[trajectory_name]
        edge_idxs = torch.tensor(trajectory['edge_idxs'][:], dtype=torch.long)
        edge_orientations = torch.tensor(trajectory['edge_orientations'][:], dtype=torch.long)
        
        edge_coordinates_data = trajectory.get('edge_coordinates', [])
        if len(edge_coordinates_data) > 0:
            edge_coordinates_np = np.array(edge_coordinates_data)
            edge_coordinates = torch.tensor(edge_coordinates_np, dtype=torch.float)
        else:
            edge_coordinates = torch.tensor([], dtype=torch.float)

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
        return len(self.keys)

    def __del__(self):
        self.trajectories.close()

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
    
"""nodes = [(0, {'pos': (0.1, 0.65)}),
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

file_path = '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic.h5'
history_len = 5
future_len = 2
edge_features = ['edge_one_hot', 'edge_orientations']
dataset = TrajectoryDataset(file_path, history_len, nodes, edges, future_len, edge_features=edge_features)

print(dataset[2])"""