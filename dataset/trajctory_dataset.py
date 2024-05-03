'''import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import h5py
import networkx as nx
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, nodes, edges, future_len=5):
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
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
        padding_length = 0
        
        if 'edge_coordinates' in trajectory:
            edge_coordinates = torch.tensor(np.array(trajectory['edge_coordinates']), dtype=torch.float)
            # Adjust coordinates based on orientation
            for i, orientation in enumerate(edge_orientations):
                if orientation == -1:
                    # Swap the coordinates for this edge
                    edge_coordinates[i] = edge_coordinates[i][[1, 0]]
        else:
            edge_coordinates = None

        total_len = self.history_len + self.future_len
        if edge_idxs.size(0) < total_len:
            padding_length = total_len - len(edge_idxs)
            edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)
            edge_orientations = torch.nn.functional.pad(edge_orientations, (0, padding_length), value=0)   
            
            # Each coordinate has shape [number_of_edges, 2, 2]
            # We need to add `padding_length` new entries, each of which is a 2x2 matrix of zeros
            zero_padding = torch.zeros((padding_length, 2, 2), dtype=torch.float)
            edge_coordinates = torch.cat((edge_coordinates, zero_padding), dim=0)

        history_indices = edge_idxs[:self.history_len]
        future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]

        history_coordinates = edge_coordinates[:self.history_len] if edge_coordinates is not None else None
        future_coordinates = edge_coordinates[self.history_len:self.history_len + self.future_len] if edge_coordinates is not None else None
        
        history_orientations = edge_orientations[:self.history_len]
        future_orientations = edge_orientations[self.history_len:self.history_len + self.future_len]

        target_mask = torch.zeros(self.future_len, dtype=torch.bool)
        if padding_length > 0:
            target_mask[:self.future_len - padding_length] = 1

        return history_indices, history_coordinates, history_orientations, future_indices, future_coordinates, future_orientations, target_mask

    def __len__(self):
        return len(self.keys)

    def __del__(self):
        self.trajectories.close()  # Properly close the HDF5 file
    
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
    
    def get_trajectory_edges_tensor(self, idx):
        """
        For a given trajectory index, returns a tensor of shape [2, num_trajectory_edges]
        where each column represents an edge involved in the trajectory and the two entries
        in each column represent the nodes connected by that edge, taking into account the edge orientation.
        """
        trajectory_name = self.keys[idx]
        trajectory = self.trajectories[trajectory_name]
        edge_idxs = torch.tensor(trajectory['edge_idxs'][:], dtype=torch.long)
        edge_orientations = torch.tensor(trajectory['edge_orientations'][:], dtype=torch.long)  # Ensure orientations are loaded

        # Fetch the corresponding edges with orientation consideration
        trajectory_edges = []
        for idx, orientation in zip(edge_idxs, edge_orientations):
            if idx == -1:  # Handle padding
                continue
            start_node, end_node = self.edges[idx]
            if orientation == -1:
                start_node, end_node = end_node, start_node  # Reverse the edge
            trajectory_edges.append((start_node, end_node))

        if trajectory_edges:
            edge_tensor = torch.tensor(trajectory_edges, dtype=torch.long).t()
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long)  # Return empty tensor if no edges
        
        return edge_tensor

    def get_n_edges(self):
        return self.graph.number_of_edges()
    

def pad_coordinates(coords_list, max_len):
    """Pad the coordinate tensors to have uniform lengths."""
    if not coords_list:
        return torch.tensor([])  # Return an empty tensor if the list is empty

    # Find out the maximum size in each dimension except the batch dimension
    max_feature_length = max(coord.shape[2] for coord in coords_list if coord is not None)  # Maximum features per edge

    padded_coords = []
    for coords in coords_list:
        if coords is not None:
            # Determine how much padding is needed in each dimension
            padding_needed_sequence = max_len - coords.size(0)
            padding_needed_features = max_feature_length - coords.size(2)

            # Apply padding to both dimensions if needed
            padded_coord = torch.nn.functional.pad(coords, (0, padding_needed_features, 0, padding_needed_sequence), "constant", 0)
            padded_coords.append(padded_coord)
        else:
            # Create a tensor of zeros with the appropriate size if no coordinates exist
            padded_coords.append(torch.zeros(max_len, 2, max_feature_length))  # Adjust the '2' if a different dimensionality is required

    # Stack the padded coordinates ensuring all have the same shape
    return torch.stack(padded_coords)

def collate_fn(batch):
    # Find the maximum length of history and future indices across all samples
    max_history_len = max(len(item[0]) for item in batch)
    max_future_len = max(len(item[2]) for item in batch)

    # Pad history and future indices
    padded_history_idxs = torch.stack([torch.nn.functional.pad(item[0], (0, max_history_len - len(item[0])), value=-1) for item in batch])
    padded_future_idxs = torch.stack([torch.nn.functional.pad(item[2], (0, max_future_len - len(item[2])), value=-1) for item in batch])
    
    # Pad masks to match the maximum length
    padded_masks = torch.stack([torch.nn.functional.pad(item[4], (0, max_future_len - len(item[4])), value=False) for item in batch])

    # Handle coordinates using the updated pad_coordinates function
    history_coordinates = pad_coordinates([item[1] for item in batch], max_history_len)
    future_coordinates = pad_coordinates([item[3] for item in batch], max_future_len)

    return padded_history_idxs, history_coordinates, padded_future_idxs, future_coordinates, padded_masks'''

import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import networkx as nx
import numpy as np
# from torch_geometric.data import Data, Batch

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
        edge_coordinates = torch.tensor(trajectory.get('edge_coordinates', []), dtype=torch.float)

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
        
        history_one_hot_edges = torch.nn.functional.one_hot(history_indices[valid_history_mask], num_classes=len(self.edges)).float()
        future_one_hot_edges = torch.nn.functional.one_hot(future_indices[valid_future_mask], num_classes=len(self.edges)).float()
        
        # Sum across the time dimension to count occurrences of each edge
        history_one_hot_edges = history_one_hot_edges.sum(dim=0)  # (num_edges,)
        future_one_hot_edges = future_one_hot_edges.sum(dim=0)  # (num_edges,)

        if 'edge_orientations' in self.edge_features:
            history_edge_features = torch.stack((history_one_hot_edges, history_edge_orientations), dim=1)
            future_edge_features = torch.stack((future_one_hot_edges, future_edge_orientations), dim=1)
        else:
            history_edge_features = history_one_hot_edges
            future_edge_features = future_one_hot_edges
        
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
            "future_edge_features": future_edge_features
        }
        
        '''return Data(x=self.node_coordinates(), edge_index=self.get_all_edges_tensor(), history_edge_attr=history_edge_features, future_edge_attr=future_edge_features,
                    history_indices=history_indices, future_indices=future_indices, history_coordinates=history_coordinates, future_coordinates=future_coordinates,
                    history_one_hot_edges=history_one_hot_edges, future_one_hot_edges=future_one_hot_edges, history_edge_orientations=history_edge_orientations, future_edge_orientations=future_edge_orientations)'''


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
        

'''def collate_fn(batch):
    # Initialize lists to store batch data
    batch_history_indices = []
    batch_future_indices = []
    batch_history_coordinates = []
    batch_future_coordinates = []
    batch_history_edge_orientations = []
    batch_future_edge_orientations = []

    # Iterate over each data item in the batch
    for data in batch:
        batch_history_indices.append(data['history_indices'])
        batch_future_indices.append(data['future_indices'])
        if data['history_coordinates'] is not None:
            batch_history_coordinates.append(data['history_coordinates'])
        if data['future_coordinates'] is not None:
            batch_future_coordinates.append(data['future_coordinates'])
        batch_history_edge_orientations.append(data['history_edge_orientations'])
        batch_future_edge_orientations.append(data['future_edge_orientations'])

    # Convert lists to tensors and reshape appropriately
    batch_history_indices = torch.stack(batch_history_indices)
    batch_future_indices = torch.stack(batch_future_indices)
    if batch_history_coordinates:
        batch_history_coordinates = torch.stack(batch_history_coordinates)
    if batch_future_coordinates:
        batch_future_coordinates = torch.stack(batch_future_coordinates)
    batch_history_edge_orientations = torch.stack(batch_history_edge_orientations)
    batch_future_edge_orientations = torch.stack(batch_future_edge_orientations)

    # Create a PyTorch Geometric batch from the list of Data objects
    graph_batch = Batch.from_data_list([data['graph_data'] for data in batch])

    # Attach reshaped data as attributes to the graph batch
    graph_batch.history_indices = batch_history_indices
    graph_batch.future_indices = batch_future_indices
    graph_batch.history_coordinates = batch_history_coordinates
    graph_batch.future_coordinates = batch_future_coordinates
    graph_batch.history_edge_orientations = batch_history_edge_orientations
    graph_batch.future_edge_orientations = batch_future_edge_orientations

    return graph_batch'''

'''def collate_fn(batch):
    return Batch.from_data_list(batch)'''

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
            "future_edge_features": future_edge_features
        }
    
nodes = [(0, {'pos': (0.1, 0.65)}),
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

'''dataset = TrajectoryDataset(file_path="/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic.h5", 
                            history_len=5, nodes=nodes, edges=edges, future_len=2, edge_features=['one_hot'])

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

for i, data in enumerate(dataloader):
    print(data["history_indices"])
    print(data["history_one_hot_edges"])
    if i == 1:
        break'''
    