'''import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import networkx as nx

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
        edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)
        edge_orientations = torch.nn.functional.pad(edge_orientations, (0, padding_length), value=0)
        
        # Pad coordinates
        if padding_length > 0 and edge_coordinates.numel() > 0:
            zero_padding = torch.zeros((padding_length, 2, 2), dtype=torch.float)
            edge_coordinates = torch.cat([edge_coordinates, zero_padding], dim=0)
        
        # Split into history and future
        history_indices = edge_idxs[:self.history_len]
        future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]
        #history_coordinates = edge_coordinates[:self.history_len] if edge_coordinates.numel() > 0 else None
        #future_coordinates = edge_coordinates[self.history_len:self.history_len + self.future_len] if edge_coordinates.numel() > 0 else None
        
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
        
        # Generate the tensor indicating nodes in history
        """node_in_history = torch.zeros((len(self.nodes), 1), dtype=torch.float)
        history_edges = [self.edges[i] for i in history_indices if i >= 0]
        history_nodes = set(node for edge in history_edges for node in edge)
        for node in history_nodes:
            node_in_history[node] = 1"""
            
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
            "history_edge_features": history_edge_features,
            "future_edge_features": future_edge_features,
            #"history_coordinates": history_coordinates,
            #"future_coordinates": future_coordinates,
            #"history_one_hot_edges": history_one_hot_edges,
            #"future_one_hot_edges": future_one_hot_edges,
            #"history_edge_orientations": history_edge_orientations,
            #"future_edge_orientations": future_edge_orientations,
            #"node_in_history": node_in_history,
        }, self.graph# , self.edges
        
    def __len__(self):
        return len(self.trajectories)

    def get_n_edges(self):
        return self.graph.number_of_edges()
    
    """def node_coordinates(self):
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
        return edge_tensor"""


def collate_fn(batch):
    graph = [item[1] for item in batch]
    # edges = [item[2] for item in batch]
    # Extract elements for each sample and stack them, handling variable lengths
    history_indices = torch.stack([item[0]['history_indices'] for item in batch])
    future_indices = torch.stack([item[0]['future_indices'] for item in batch])
    
    #history_one_hot_edges = torch.stack([item[0]['history_one_hot_edges'] for item in batch])
    #future_one_hot_edges = torch.stack([item[0]['future_one_hot_edges'] for item in batch])

    # Coordinates
    #history_coordinates = [item[0]['history_coordinates'] for item in batch if item[0]['history_coordinates'] is not None]
    #future_coordinates = [item[0]['future_coordinates'] for item in batch if item[0]['future_coordinates'] is not None]
    
    #history_edge_orientations = torch.stack([item[0]['history_edge_orientations'] for item in batch])
    #future_edge_orientations = torch.stack([item[0]['future_edge_orientations'] for item in batch])
    
    history_edge_features = torch.stack([item[0]['history_edge_features'] for item in batch])
    future_edge_features = torch.stack([item[0]['future_edge_features'] for item in batch])
    
    #history_one_hot_nodes = torch.stack([item[0]['node_in_history'] for item in batch])
    
    # Stack coordinates if not empty
    """if history_coordinates:
        history_coordinates = torch.stack(history_coordinates)
    if future_coordinates:
        future_coordinates = torch.stack(future_coordinates)"""

    return {
            "history_indices": history_indices,
            "future_indices": future_indices,
            #"history_coordinates": history_coordinates,
            #"future_coordinates": future_coordinates,
            #"history_one_hot_edges": history_one_hot_edges,
            #"future_one_hot_edges": future_one_hot_edges,
            #"history_edge_orientations": history_edge_orientations,
            #"future_edge_orientations": future_edge_orientations,
            "history_edge_features": history_edge_features,
            "future_edge_features": future_edge_features,
            #"history_one_hot_nodes": history_one_hot_nodes,
            "graph": graph,
            # "edges": edges,
        }'''
        
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import networkx as nx
import numpy as np
import time
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, edge_features=None, device=None):
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.device = device
        self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features = 5
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features = 6
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(file_path, self.device)
        
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float64, device=self.device)
        
    @staticmethod
    def load_new_format(file_path, device):
        paths = []
        with h5py.File(file_path, 'r') as new_hf:
            node_coordinates = torch.tensor(new_hf['graph']['node_coordinates'][:], dtype=torch.float, device=device)
            # Normalize the coordinates to (0, 1) if any of the coordinates is larger than 1
            if node_coordinates.max() > 1:
                max_values = node_coordinates.max(0)[0]
                min_values = node_coordinates.min(0)[0]
                node_coordinates[:, 0] = (node_coordinates[:, 0] - min_values[0]) / (max_values[0] - min_values[0])
                node_coordinates[:, 1] = (node_coordinates[:, 1] - min_values[1]) / (max_values[1] - min_values[1])
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
    
    # @staticmethod
    def build_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in indexed_edges:
            graph.add_edge(start, end, index=index, default_orientation=(start, end))
        return graph

    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
    
        edge_idxs = trajectory['edge_idxs']
        if 'edge_orientations' in self.edge_features:
            edge_orientations = trajectory['edge_orientations']
        
        # Calculate the required padding length
        total_len = self.history_len + self.future_len
        padding_length = max(total_len - len(edge_idxs), 0)
        
        # Pad edge indices, orientations, and coordinates
        edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)
        if 'edge_orientations' in self.edge_features:
            edge_orientations = torch.nn.functional.pad(edge_orientations, (0, padding_length), value=0)
        
        # Split into history and future
        history_indices = edge_idxs[:self.history_len]
        future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]

        # Extract and generate features
        history_edge_features, future_edge_features = self.generate_edge_features(history_indices, future_indices, self.edge_coordinates)

        return {
            "history_indices": history_indices,
            "future_indices": future_indices,
            "history_edge_features": history_edge_features,
            "future_edge_features": future_edge_features,
        }

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
            future_edge_features = torch.cat((future_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
            pass
        if 'edge_orientations' in self.edge_features:
            history_edge_features = torch.cat((history_edge_features, history_edge_orientations.float()), dim=1)
            future_edge_features = torch.cat((future_edge_features, future_edge_orientations.float()), dim=1)
        return history_edge_features, future_edge_features
        
    def __len__(self):
        return len(self.trajectories)


def collate_fn(batch):
    history_indices = torch.stack([item['history_indices'] for item in batch])
    future_indices = torch.stack([item['future_indices'] for item in batch])
    history_edge_features = torch.stack([item['history_edge_features'] for item in batch])
    future_edge_features = torch.stack([item['future_edge_features'] for item in batch])

    return {
        "history_indices": history_indices,
        "future_indices": future_indices,
        "history_edge_features": history_edge_features,
        "future_edge_features": future_edge_features,
    }
