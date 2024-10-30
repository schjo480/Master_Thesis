import torch
from torch_geometric.data import Dataset, Data
import h5py
import numpy as np
from tqdm import tqdm
import networkx as nx

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        # Specify that 'binary_future' should not be concatenated across the zeroth dimension
        if (key == 'y') or (key == 'history_indices') or (key == 'future_indices'):
            return None  # This will add a new batch dimension during batching
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)  # Default behaviour

class TrajectoryGeoDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, edge_features=None, device=None, embedding_dim=None, conditional_future_len=None):
        super().__init__()
        self.ct = 0
        self.file_path = file_path
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.embedding_dim = embedding_dim
        self.device = device
        self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(self.file_path, self.device)
        
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float64, device=self.device)
        self.edge_index = self._build_edge_index()
        self.longest_future = max([len(traj['edge_idxs']) for traj in self.trajectories]) - self.history_len
        self.avg_future_len = sum([len(traj['edge_idxs']) for traj in self.trajectories]) / len(self.trajectories) - self.history_len
        if self.future_len <= 0:
            self.future_len = self.longest_future
        self.conditional_future_len = conditional_future_len

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
    
    def _build_edge_index(self):
        """
        Builds the edge index for the line graph.
        Returns:
            torch.Tensor: The edge index for the line graph with dimensions [2, num_edges].
        """
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
        if self.conditional_future_len is not None:
            traj_len = self.conditional_future_len
        else:
            traj_len = len(edge_idxs) - self.history_len
        
        padding_length = max(self.history_len + self.future_len - len(edge_idxs), 0)
        edge_idxs = torch.nn.functional.pad(edge_idxs, (0, padding_length), value=-1)

        # Split into history and future
        if 'start_end' in self.edge_features:
            history_indices = torch.cat((torch.tensor([edge_idxs[0]], device=self.device), torch.tensor([edge_idxs[edge_idxs >= 0][-1]], device=self.device)))
            future_indices = edge_idxs[1:-2]
            future_indices_check = future_indices[future_indices >= 0]
        else:
            history_indices = edge_idxs[:self.history_len]
            future_indices = edge_idxs[self.history_len:self.history_len + self.future_len]
            future_indices_check = future_indices[future_indices >= 0]
        
        # Extract and generate features
        history_edge_features, binary_future = self.generate_edge_features(history_indices, future_indices, traj_len, edge_idxs)
        data = MyData(x=history_edge_features,          # (batch_size * num_edges, num_edge_features)
                    edge_index=self.edge_index,         # (2, num_edges)
                    y=binary_future,                    # (batch_size, num_edges, 1)
                    history_indices=history_indices,    # (batch_size, history_len)
                    future_indices=future_indices,      # (batch_size, future_len)
                    num_nodes=self.num_edges)
        
        return data

    def generate_edge_features(self, history_indices, future_indices, traj_len, edge_idxs):
        """
        Generate edge features based on the given history indices, future indices, trajectory length, and edge indices.
        Args:
            history_indices (torch.Tensor): Tensor containing the indices of the history edges.
            future_indices (torch.Tensor): Tensor containing the indices of the future edges.
            traj_len (int): Length of the trajectory.
            edge_idxs (torch.Tensor): Tensor containing the indices of the edges.
        Returns:
            history_edge_features (torch.Tensor): Tensor containing the generated history edge features with dimension [num_edges, num_edge_features].
            binary_future (torch.Tensor): Tensor containing the binarized future edges with dimension [num_edges, 1].
        """
        
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
        binary_future = future_one_hot_edges.view(-1, 1).float()
        if 'start_end' in self.edge_features:
            start_edge = edge_idxs[0]
            end_edge = edge_idxs[edge_idxs >= 0][-1]
            start_node, end_node = self.find_trajectory_endpoints(edge_idxs[edge_idxs >= 0])
            history_edge_features = torch.cat((history_edge_features, torch.zeros_like(binary_future)), dim=1)
            if self.edge_coordinates[start_edge][0][0] == start_node[0] and self.edge_coordinates[start_edge][0][1] == start_node[1]:
                history_edge_features[start_edge, 0] = 1
                history_edge_features[start_edge, 1] = 1
            else:
                history_edge_features[start_edge, 0] = 1
                history_edge_features[start_edge, 1] = -1
            if self.edge_coordinates[start_edge][1][0] == end_node[0] and self.edge_coordinates[start_edge][1][1] == end_node[1]:
                history_edge_features[end_edge, 0] = 1
                history_edge_features[end_edge, 1] = 1
            else:
                history_edge_features[end_edge, 0] = 1    
                history_edge_features[end_edge, 1] = -1
                
                    
            if 'coordinates' in self.edge_features:
                if 'munich' in self.file_path or 'tdrive' in self.file_path or 'geolife' in self.file_path:
                    # Get the coordinates of the history edges
                    history_edge_coords = self.edge_coordinates[history_indices[valid_history_mask]]  # Shape: (num_valid_history_edges, 2, coordinate_dim)

                    # Compute midpoints of the history edges
                    history_edge_midpoints = (history_edge_coords[:, 0, :] + history_edge_coords[:, 1, :]) / 2  # Shape: (num_valid_history_edges, coordinate_dim)

                    # Calculate the mean coordinate
                    mean_coordinate = history_edge_midpoints.mean(dim=0)  # Shape: (coordinate_dim,)

                    # **Recenter All Edge Coordinates**
                    mean_coordinate = mean_coordinate.view(1, 1, -1)  # Reshape to (1, 1, coordinate_dim) for broadcasting
                    recentered_edge_coordinates = self.edge_coordinates - mean_coordinate  # Shape: (num_edges, 2, coordinate_dim)

                    # **Normalize the Recentered Coordinates to 0-1 Range**
                    # Flatten the recentered coordinates
                    recentered_edge_coords_flat = recentered_edge_coordinates.view(len(self.edges), -1)  # Shape: (num_edges, 2 * coordinate_dim)

                    history_edge_features = torch.cat((history_edge_features, recentered_edge_coords_flat.float()), dim=1)

                else:
                    history_edge_features = torch.cat((history_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
                
            if 'pw_distance' in self.edge_features:
                start_edge_coords = self.edge_coordinates[start_edge]
                end_edge_coords = self.edge_coordinates[end_edge]
                edge_middles = (self.edge_coordinates[:, 0, :] + self.edge_coordinates[:, 1, :]) / 2
                start_edge_middle = (start_edge_coords[0, :] + start_edge_coords[1, :]) / 2
                end_edge_middle = (end_edge_coords[0, :] + end_edge_coords[1, :]) / 2
                start_distances = torch.norm(edge_middles - start_edge_middle, dim=1, keepdim=True)
                end_distances = torch.norm(edge_middles - end_edge_middle, dim=1, keepdim=True)
                history_edge_features = torch.cat((history_edge_features, start_distances.float(), end_distances.float()), dim=1)
            if 'edge_length' in self.edge_features:
                edge_lengths = torch.norm(self.edge_coordinates[:, 0, :] - self.edge_coordinates[:, 1, :], dim=1, keepdim=True)
                history_edge_features = torch.cat((history_edge_features, edge_lengths.float()), dim=1)
            if 'edge_angles' in self.edge_features:
                start, end = self.find_trajectory_endpoints(edge_idxs[edge_idxs >= 0])
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

        else:
            if 'coordinates' in self.edge_features:
                if 'munich' in self.file_path or 'tdrive' in self.file_path or 'geolife' in self.file_path:
                    # Get the coordinates of the history edges
                    history_edge_coords = self.edge_coordinates[history_indices[valid_history_mask]]  # Shape: (num_valid_history_edges, 2, coordinate_dim)

                    # Compute midpoints of the history edges
                    history_edge_midpoints = (history_edge_coords[:, 0, :] + history_edge_coords[:, 1, :]) / 2  # Shape: (num_valid_history_edges, coordinate_dim)

                    # Calculate the mean coordinate
                    mean_coordinate = history_edge_midpoints.mean(dim=0)  # Shape: (coordinate_dim,)

                    # **Recenter All Edge Coordinates**
                    mean_coordinate = mean_coordinate.view(1, 1, -1)  # Reshape to (1, 1, coordinate_dim) for broadcasting
                    recentered_edge_coordinates = self.edge_coordinates - mean_coordinate  # Shape: (num_edges, 2, coordinate_dim)

                    # **Normalize the Recentered Coordinates to 0-1 Range**
                    # Flatten the recentered coordinates
                    recentered_edge_coords_flat = recentered_edge_coordinates.view(len(self.edges), -1)  # Shape: (num_edges, 2 * coordinate_dim)

                    history_edge_features = torch.cat((history_edge_features, recentered_edge_coords_flat.float()), dim=1)

                else:
                    history_edge_features = torch.cat((history_edge_features, torch.flatten(self.edge_coordinates, start_dim=1).float()), dim=1)
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
            if 'future_len' in self.edge_features:
                future_len_feature = torch.tensor([traj_len / self.longest_future], device=self.device).float().repeat(self.num_edges).unsqueeze(1)
                history_edge_features = torch.cat((history_edge_features, future_len_feature), dim=1)
                
        history_edge_features = torch.cat((history_edge_features, torch.zeros_like(binary_future)), dim=1)
        history_edge_features = torch.nan_to_num(history_edge_features, nan=0.0)
        if 'one_hot_edges' not in self.edge_features:
            history_edge_features = history_edge_features[:, 1:]
        
        return history_edge_features, binary_future
    
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
        """
        Builds a graph using the nodes and edges provided.
        Returns:
            graph (nx.Graph): The constructed graph.
        """
        
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        self.indexed_edges = [((start, end), index) for index, (start, end) in enumerate(self.edges)]
        for (start, end), index in self.indexed_edges:
            graph.add_edge(start, end, index=index, default_orientation=(start, end))
        return graph
    
def custom_collate_fn(batch):
    """
    Custom collate function for batching graph data objects in PyTorch Geometric.

    This function combines a list of `MyData` graph objects into a single batched graph, adjusting node and edge indices appropriately to prevent overlap between individual graphs in the batch.

    Args:
        batch (list of MyData): A list containing `MyData` objects. Each `MyData` object represents a graph with the following attributes:
            - `x` (torch.Tensor): Node feature matrix of shape `(num_nodes, feature_dim)`.
            - `edge_index` (torch.Tensor): Edge indices of shape `(2, num_edges)`.
            - `y` (torch.Tensor): Target values.
            - `history_indices` (torch.Tensor): Indices representing historical data.
            - `future_indices` (torch.Tensor): Indices representing future data.
            - `num_nodes` (int): Number of nodes in the graph (optional; can be inferred).

    Returns:
        MyData: A single `MyData` object containing the batched graph data with the following attributes:
            - `x` (torch.Tensor): Concatenated node features from all graphs in the batch, shape `(batch_size * num_nodes, feature_dim)`.
            - `edge_index` (torch.Tensor): Adjusted and concatenated edge indices from all graphs in the batch, shape `(2, total_num_edges)`.
            - `y` (torch.Tensor): Concatenated target values from all graphs in the batch.
            - `history_indices` (torch.Tensor): Concatenated history indices from all graphs in the batch.
            - `future_indices` (torch.Tensor): Concatenated future indices from all graphs in the batch.
            - `num_nodes` (int): Total number of nodes in the batched graph (`batch_size * num_nodes_per_graph`).
    """
    batch_size = len(batch)
    num_edges = batch[0].num_nodes

    x = torch.cat([data.x for data in batch], dim=0)
    y = torch.cat([data.y for data in batch], dim=0)
    history_indices = torch.cat([data.history_indices for data in batch], dim=0)
    future_indices = torch.cat([data.future_indices for data in batch], dim=0)

    # Create a new edge index for each item in the batch
    edge_indices = [data.edge_index for data in batch]
    for i, edge_index in enumerate(edge_indices):
        edge_indices[i] = edge_index + i * num_edges

    edge_index = torch.cat(edge_indices, dim=1)
    num_nodes = batch_size * num_edges

    return MyData(x=x, edge_index=edge_index, y=y, history_indices=history_indices, future_indices=future_indices, num_nodes=num_nodes)
