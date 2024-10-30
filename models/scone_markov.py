import numpy as np
import networkx as nx
import h5py
from tqdm import tqdm
from collections import deque
import sys
sys.path.append('/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction')
from dataset.trajectory_dataset_rnn import TrajectoryDataset

class Markov_Model():
    def __init__(self, history_len, future_len, edges, node_positions, use_padding=False):
        """
        Initializes the Markov Model.
        :param history_len: Fixed number of prior states (nodes) to consider.
        :param future_len: Default number of future nodes to predict.
        :param edges: List of edges in the graph.
        :param node_positions: Dictionary mapping node IDs to their coordinates.
        :param use_padding: Boolean indicating whether to pad shorter paths with stop tokens.
        """
        self.history_len = history_len
        self.future_len = future_len
        self.edges = edges
        self.node_positions = node_positions
        self.use_padding = use_padding
        self.weights = {}
        # Determine the number of nodes
        self.num_nodes = max(node_positions.keys()) + 1  # Assuming nodes are indexed from 0
        # Define the stop token
        self.stop_token = self.num_nodes  # Equal to num_nodes
        self.unseen_prefix_count = 0  # For tracking unseen prefixes during prediction (optional)

    def train(self, G, paths):
        """
        Trains the model on given paths in graph G.
        """
        self.G = G  # Store the graph for use in prediction
        path_edges = [list(path['edge_idxs']) for path in paths]
        paths = self.convert_edge_indices_to_node_paths(path_edges)
        self.weights = {}
        for path in paths:
            # Pad the path to length history_len + future_len if necessary
            if self.use_padding and len(path) < self.history_len + self.future_len:
                total_len = self.history_len + self.future_len
                padding_len = total_len - len(path)
                path = path + [self.stop_token] * padding_len
            # Generate training data
            for i in range(len(path) - self.history_len):
                prefix = tuple(path[i:i + self.history_len])
                next_node = path[i + self.history_len]
                if prefix not in self.weights:
                    self.weights[prefix] = {}
                if next_node not in self.weights[prefix]:
                    self.weights[prefix][next_node] = 0
                self.weights[prefix][next_node] += 1
        # For each prefix, add all neighbors of the last node in the prefix and the stop token
        for prefix in self.weights:
            last_node = prefix[-1]
            if last_node != self.stop_token and last_node in self.G:
                neighbors = list(self.G.neighbors(last_node))
            else:
                neighbors = []
            possible_next_nodes = neighbors + [self.stop_token]
            for neighbor in possible_next_nodes:
                if neighbor not in self.weights[prefix]:
                    self.weights[prefix][neighbor] = 0
        # Normalize the weights
        for prefix in self.weights:
            total = sum(self.weights[prefix].values())
            if total > 0:
                for node in self.weights[prefix]:
                    self.weights[prefix][node] /= total
            else:
                # All counts are zero, assign uniform probability
                num_nodes = len(self.weights[prefix])
                for node in self.weights[prefix]:
                    self.weights[prefix][node] = 1.0 / num_nodes

    def predict(self, prefix, max_future_len=None):
        """
        Predicts the next nodes based on the given prefix.
        :param prefix: Tuple of node states representing the current state.
        :param max_future_len: Maximum number of future nodes to predict. If None, uses self.future_len.
        :return: List of predicted nodes.
        """
        if max_future_len is None:
            max_future_len = self.future_len
        predictions = []
        current_prefix = prefix
        for _ in range(max_future_len):
            if tuple(current_prefix) in self.weights:
                next_nodes = list(self.weights[tuple(current_prefix)].keys())
                probabilities = list(self.weights[tuple(current_prefix)].values())
                next_node = np.random.choice(next_nodes, p=probabilities)
                predictions.append(next_node)
                current_prefix = current_prefix[1:] + (next_node,)
                if next_node == self.stop_token:
                    break  # Stop predicting if stop token is predicted
            else:
                # If the prefix is unseen, generate possible next nodes
                last_node = current_prefix[-1]
                if last_node != self.stop_token and last_node in self.G:
                    neighbors = list(self.G.neighbors(last_node))
                else:
                    neighbors = []
                possible_next_nodes = neighbors + [self.stop_token]
                num_nodes = len(possible_next_nodes)
                probabilities = [1.0 / num_nodes] * num_nodes  # Uniform probabilities
                next_node = np.random.choice(possible_next_nodes, p=probabilities)
                predictions.append(next_node)
                current_prefix = current_prefix[1:] + (next_node,)
                self.unseen_prefix_count += 1  # Optional tracking
                if next_node == self.stop_token:
                    break  # Stop predicting if stop token is predicted
        return predictions

    def test(self, test_paths):
        """
        Evaluates the model's predictions on test paths.
        """
        correct_sequence_predictions = 0
        total_sequences = 0
        correct_node_predictions = 0
        total_node_predictions = 0

        total_distance = 0  # For calculating ADE
        total_ade_steps = 0  # Number of steps used in ADE
        total_final_distance = 0  # For calculating FDE
        total_fde_sequences = 0   # Number of sequences used in FDE

        path_edges = [list(path['edge_idxs']) for path in test_paths]
        test_paths = self.convert_edge_indices_to_node_paths(path_edges)
        for path in test_paths:
            # Determine the actual future length for this path
            actual_future_len = len(path) - self.history_len
            if self.use_padding and actual_future_len < self.future_len:
                total_len = self.history_len + self.future_len
                padding_len = total_len - len(path)
                path = path + [self.stop_token] * padding_len
                actual_future_len = self.future_len
            elif actual_future_len > self.future_len:
                actual_future_len = self.future_len

            # Check if the path is long enough
            if len(path) >= self.history_len + actual_future_len:
                prefix = tuple(path[:self.history_len])
                actual_next_nodes = path[self.history_len:self.history_len + actual_future_len]
                # Predict only the required number of future nodes
                predicted_next_nodes = self.predict(prefix, max_future_len=actual_future_len)
                # Ensure the predicted list is the same length as actual
                predicted_next_nodes = predicted_next_nodes[:len(actual_next_nodes)]
                # Update total sequences
                total_sequences += 1
                # Node-wise accuracy
                total_node_predictions += len(predicted_next_nodes)
                for pred_node, actual_node in zip(predicted_next_nodes, actual_next_nodes):
                    if pred_node == actual_node:
                        correct_node_predictions += 1
                # Sequence accuracy
                if predicted_next_nodes == actual_next_nodes:
                    correct_sequence_predictions += 1
                # Remove stop tokens from predictions and ground truth
                predicted_next_nodes_clipped = [node for node in predicted_next_nodes if node != self.stop_token]
                actual_next_nodes_clipped = [node for node in actual_next_nodes if node != self.stop_token]
                len_pred = len(predicted_next_nodes_clipped)
                len_actual = len(actual_next_nodes_clipped)
                # Handle empty sequences after clipping
                if len_actual == 0 and len_pred == 0:
                    # Both sequences are empty after clipping; nothing to compute for ADE and FDE
                    continue
                elif len_actual == 0:
                    # No actual future nodes; compare predicted nodes to the last position in prefix
                    pred_pos_list = [self.node_positions.get(node) for node in predicted_next_nodes_clipped]
                    actual_pos = self.node_positions.get(prefix[-1])
                    for pred_pos in pred_pos_list:
                        if pred_pos is not None and actual_pos is not None:
                            distance = np.linalg.norm(pred_pos - actual_pos)
                            total_distance += distance
                    total_ade_steps += len_pred
                    # For FDE, compare last predicted position to last position in prefix
                    if pred_pos_list:
                        final_pred_pos = pred_pos_list[-1]
                        if final_pred_pos is not None and actual_pos is not None:
                            final_distance = np.linalg.norm(final_pred_pos - actual_pos)
                            total_final_distance += final_distance
                            total_fde_sequences += 1
                    continue
                elif len_pred == 0:
                    # No predicted future nodes; compare actual future nodes to last position in prefix
                    actual_pos_list = [self.node_positions.get(node) for node in actual_next_nodes_clipped]
                    pred_pos = self.node_positions.get(prefix[-1])
                    for actual_pos in actual_pos_list:
                        if pred_pos is not None and actual_pos is not None:
                            distance = np.linalg.norm(pred_pos - actual_pos)
                            total_distance += distance
                    total_ade_steps += len_actual
                    # For FDE, compare last actual position to last position in prefix
                    if actual_pos_list:
                        final_actual_pos = actual_pos_list[-1]
                        if pred_pos is not None and final_actual_pos is not None:
                            final_distance = np.linalg.norm(pred_pos - final_actual_pos)
                            total_final_distance += final_distance
                            total_fde_sequences += 1
                    continue
                else:
                    # Both predicted and actual sequences have nodes after clipping
                    # For ADE
                    sequence_total_distance = 0
                    min_len = min(len_pred, len_actual)
                    for i in range(min_len):
                        pred_node = predicted_next_nodes_clipped[i]
                        actual_node = actual_next_nodes_clipped[i]
                        pred_pos = self.node_positions.get(pred_node)
                        actual_pos = self.node_positions.get(actual_node)
                        if pred_pos is not None and actual_pos is not None:
                            distance = np.linalg.norm(pred_pos - actual_pos)
                            sequence_total_distance += distance
                    # Handle extra nodes in the longer sequence
                    if len_pred > len_actual:
                        # Predictions are longer
                        last_actual_node = actual_next_nodes_clipped[-1]
                        actual_pos = self.node_positions.get(last_actual_node)
                        for i in range(len_actual, len_pred):
                            pred_node = predicted_next_nodes_clipped[i]
                            pred_pos = self.node_positions.get(pred_node)
                            if pred_pos is not None and actual_pos is not None:
                                distance = np.linalg.norm(pred_pos - actual_pos)
                                sequence_total_distance += distance
                        total_ade_steps += len_pred
                    elif len_actual > len_pred:
                        # Actual is longer
                        last_pred_node = predicted_next_nodes_clipped[-1]
                        pred_pos = self.node_positions.get(last_pred_node)
                        for i in range(len_pred, len_actual):
                            actual_node = actual_next_nodes_clipped[i]
                            actual_pos = self.node_positions.get(actual_node)
                            if pred_pos is not None and actual_pos is not None:
                                distance = np.linalg.norm(pred_pos - actual_pos)
                                sequence_total_distance += distance
                        total_ade_steps += len_actual
                    else:
                        total_ade_steps += len_actual  # or len_pred
                    total_distance += sequence_total_distance
                    # For FDE
                    final_pred_node = predicted_next_nodes_clipped[-1]
                    final_actual_node = actual_next_nodes_clipped[-1]
                    final_pred_pos = self.node_positions.get(final_pred_node)
                    final_actual_pos = self.node_positions.get(final_actual_node)
                    if final_pred_pos is not None and final_actual_pos is not None:
                        final_distance = np.linalg.norm(final_pred_pos - final_actual_pos)
                        total_final_distance += final_distance
                        total_fde_sequences += 1
            else:
                # Path is too short; skip it or handle accordingly
                if not self.use_padding:
                    continue
                else:
                    # Pad the path if use_padding is True
                    pass  # Already padded above

        # Calculate metrics
        if total_sequences > 0 and total_node_predictions > 0:
            sequence_accuracy = correct_sequence_predictions / total_sequences
            node_accuracy = correct_node_predictions / total_node_predictions
        else:
            sequence_accuracy = 0
            node_accuracy = 0

        if total_ade_steps > 0:
            ade = total_distance / total_ade_steps
        else:
            ade = 0

        if total_fde_sequences > 0:
            fde = total_final_distance / total_fde_sequences
        else:
            fde = 0

        return sequence_accuracy, node_accuracy, ade, fde

    def convert_edge_indices_to_node_paths(self, path_edges):
        """
        Converts paths of edge indices to paths of node indices, handling potential reversal of the first edge.

        :param path_edges: List of paths where each path is a list of edge indices
        :return: List of paths with node indices
        """
        node_paths = []
        for path in path_edges:
            if not path:
                continue
            
            node_path = []
            # Determine the starting node
            if len(path) > 1:
                # Use the second edge to determine the orientation of the first edge
                first_edge = self.edges[path[0]]
                second_edge = self.edges[path[1]]
                if first_edge[1] == second_edge[0] or first_edge[1] == second_edge[1]:
                    # Normal orientation
                    start_node = first_edge[0]
                else:
                    # Reversed orientation
                    start_node = first_edge[1]
            else:
                # Only one edge in the path, arbitrarily choose the first node
                start_node = self.edges[path[0]][0]

            node_path.append(start_node)  # Start with the determined starting node

            # Initialize the last node added to the path
            last_node = start_node

            # Process each edge in the path
            for edge_idx in path:
                edge = self.edges[edge_idx]
                if edge[0] == last_node:
                    # If the first node of the current edge matches the last added node
                    next_node = edge[1]
                else:
                    # If the second node of the current edge matches the last added node (or doesn't match due to incorrect edge order)
                    next_node = edge[0]

                node_path.append(next_node)
                last_node = next_node  # Update the last node added to the path

            node_paths.append(node_path)

        return node_paths


# === Define Dataset and Model parameters ===
dataset = 'geolife'
history_len = 6
future_len = 100
paths, nodes, edges, edge_coordinates = TrajectoryDataset.load_new_format(f'/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/{dataset}_train.h5', device='cpu')
node_positions = {node_id: data['pos'] for node_id, data in nodes}
# Build Graph
G = nx.Graph()
G.add_nodes_from(nodes)
indexed_edges = [((start, end), index) for index, (start, end) in enumerate(edges)]
for (start, end), index in indexed_edges:
    G.add_edge(start, end, index=index, default_orientation=(start, end))


markov = Markov_Model(
    history_len=history_len,
    future_len=future_len,
    edges=edges,
    node_positions=node_positions,
    use_padding=True,  # Enable padding
)
markov.train(G, paths)
val_paths, test_nodes, val_edges, val_edge_coordinates = TrajectoryDataset.load_new_format(f'/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/{dataset}_val.h5', device='cpu')
test_paths, test_nodes, test_edges, test_edge_coordinates = TrajectoryDataset.load_new_format(f'/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/{dataset}_test.h5', device='cpu')

sequence_accuracy, accuracy, ade, fde = markov.test(paths)
#print("Train Sequence Accuracy:", sequence_accuracy)
print("Train Node Accuracy:", accuracy)
print("Train Average Displacement Error (ADE):", ade)
print("Train Final Displacement Error (FDE):", fde)
print("\n")
val_sequence_accuracy, val_accuracy, val_ade, val_fde = markov.test(val_paths)
#print("Validation Sequence Accuracy:", val_sequence_accuracy)
print("Validation Node Accuracy:", val_accuracy)
print("Validation Average Displacement Error (ADE):", val_ade)
print("Validation Final Displacement Error (FDE):", val_fde)
print("\n")
test_sequence_accuracy, test_accuracy, test_ade, test_fde = markov.test(test_paths)
#print("Test Sequence Accuracy:", test_sequence_accuracy)
print("Test Node Accuracy:", test_accuracy)
print("Test Average Displacement Error (ADE):", test_ade)
print("Test Final Displacement Error (FDE):", test_fde)
