import torch
from torch_geometric.data import Dataset, Data
import h5py
import numpy as np
from tqdm import tqdm

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
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.embedding_dim = embedding_dim
        self.device = device
        '''self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features += 4
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features += 1'''
        '''if 'pos_encoding' in self.edge_features:
            self.num_edge_features += self.embedding_dim'''
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(self.file_path, self.device)
        
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float64, device=self.device)
        '''self.positional_encoding = self.generate_positional_encodings().float()'''
        self.edge_index = self._build_edge_index()
        # self.num_edges = self.edge_index.size(1)

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
        
        return history_edge_features, future_edge_features
    
    def build_graph(self):
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in indexed_edges:
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
