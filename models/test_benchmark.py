import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import h5py
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import wandb
import os

class AlternativeAutoregressiveTrajectoryDataset(Dataset):
    def __init__(self, file_path, history_len, future_len, dataset, edge_features=None, device=None):
        self.device = device
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        self.dataset = dataset
        if 'road_type' in self.edge_features:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates, self.road_type = self.load_new_format(self.file_path, self.edge_features, self.device)
            self.road_type = torch.tensor(self.road_type, dtype=torch.float64, device=self.device)
            self.num_road_types = self.road_type.size(1)
        else:
            self.trajectories, self.nodes, self.edges, self.edge_coordinates = self.load_new_format(file_path, self.edge_features, self.device)
        self.edge_coordinates = torch.tensor(self.edge_coordinates, dtype=torch.float32, device=self.device)
        self.edge_coordinates = torch.cat((self.edge_coordinates, torch.zeros((1, 2, 2), device=self.device)), dim=0)
        
        self.longest_traj = max([len(traj['edge_idxs']) for traj in self.trajectories])
        
        self.num_edges = len(self.edges)
        self.stop_token = torch.tensor(self.num_edges)
        self.graph = self.build_graph()
        self.edge_list = list(self.graph.edges())
        self.data = self.prepare_data()
    
    def prepare_data(self):
        data = []
        # Preprocess and move data to the specified device
        if self.dataset == 'train':
            for trajectory in self.trajectories:
                trajectory['edge_idxs'] = torch.cat((trajectory['edge_idxs'], torch.tensor([self.stop_token])), dim=0)
                for i in range(1, self.longest_traj+1):
                    sequence = trajectory['edge_idxs'][:i]
                    if i < len(trajectory['edge_idxs']):
                        target = trajectory['edge_idxs'][i]
                    else:
                        target = self.stop_token
                                        
                    feature_tensor = self.generate_edge_features(sequence, target)  # ()

                    # Create zero-filled tensors for target
                    target_tensor_bin = torch.zeros(self.num_edges+1, device=self.device)
                    target_tensor_bin[target] = 1

                    # Generate mask based on the last edge in the sequence
                    mask = self.create_mask(sequence)

                    target_tensor = torch.tensor(target, dtype=torch.long)
                    data.append((feature_tensor, target_tensor, target_tensor_bin, mask))
                    
        elif self.dataset in ['val', 'test']:
            for trajectory in self.trajectories:
                # Fixed input length (history_len) for evaluation
                if len(trajectory['edge_idxs']) < self.history_len:
                    continue  # Skip sequences shorter than history_len
                
                input_sequence = trajectory['edge_idxs'][:self.history_len]
                output_sequence = trajectory['edge_idxs'][self.history_len:self.history_len + self.future_len]
                
                # Append stop token to the output sequence if it is shorter than future_len
                if len(output_sequence) < self.future_len:
                    output_sequence = torch.cat((output_sequence, torch.tensor([self.stop_token] * (self.future_len - len(output_sequence)), device=self.device)))
                
                feature_tensor = self.generate_edge_features(input_sequence, output_sequence)
                
                # Generate output binary vector
                target_tensor_bin = torch.zeros(self.future_len, self.num_edges + 1, device=self.device)
                for j, target in enumerate(output_sequence):
                    target_tensor_bin[j][target] = 1
                
                # Mask for the final edge in the sequence
                mask = self.create_mask(input_sequence)

                # The target tensor for the evaluation could be the actual sequence or the one-hot version
                data.append((feature_tensor, output_sequence, target_tensor_bin, mask))
        
        return data
    
    def create_mask(self, sequence):
        # Create mask based on the last edge in the sequence
        mask = torch.zeros(self.num_edges + 1, device=self.device)
        if sequence is not None:
            last_edge_index = sequence[-1]
            if last_edge_index != self.num_edges:  # Exclude the stop token
                last_edge = self.edge_list[last_edge_index]
                neighbors = [idx for idx, edge in enumerate(self.edge_list)
                             if edge[0] == last_edge[0] or edge[1] == last_edge[1]
                             or edge[0] == last_edge[1] or edge[1] == last_edge[0]]
                mask[neighbors] = 1
                mask[last_edge_index] = 0
                if len(sequence) > 1:
                    second_last_edge_index = sequence[-2]
                    mask[second_last_edge_index] = 0  # Ensure the current edge is not included in the mask
        return mask
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Directly return the preprocessed data, which is already on the correct device
        return self.data[idx]
    
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
            
    def generate_edge_features(self, sequence, target):
        feature_tensor = sequence.unsqueeze(1)
        
        if 'coordinates' in self.edge_features:
            feature_tensor = torch.cat((feature_tensor, torch.flatten(self.edge_coordinates[sequence], start_dim=1)), dim=1)
        if 'road_type' in self.edge_features:
            feature_tensor = torch.cat((feature_tensor, self.road_type.float()), dim=1)
        if 'pw_distance' in self.edge_features:
            last_history_edge_coords = self.edge_coordinates[sequence[-1]]
            edge_middles = (self.edge_coordinates[sequence, 0, :] + self.edge_coordinates[sequence, 1, :]) / 2
            last_history_edge_middle = (last_history_edge_coords[0, :] + last_history_edge_coords[1, :]) / 2  # Last edge in the history_indices
            distances = torch.norm(edge_middles - last_history_edge_middle, dim=1, keepdim=True)
            feature_tensor = torch.cat((feature_tensor, distances.float()), dim=1)
        if 'edge_length' in self.edge_features:
            edge_lengths = torch.norm(self.edge_coordinates[sequence, 0, :] - self.edge_coordinates[sequence, 1, :], dim=1, keepdim=True)
            feature_tensor = torch.cat((feature_tensor, edge_lengths.float()), dim=1)
        if 'edge_angles' in self.edge_features:
            start, end = self.find_trajectory_endpoints(sequence, target)
            v1 = end - start
            starts = self.edge_coordinates[sequence, 0, :]
            ends = self.edge_coordinates[sequence, 1, :]
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
            feature_tensor = torch.cat((feature_tensor, cosines.float()), dim=1)

        return feature_tensor
    
    def find_trajectory_endpoints(self, edge_sequence, target):
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
        if len(edge_sequence) <= 1:
            edge_sequence = torch.cat((edge_sequence, torch.tensor([target[0]])), dim=0)
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
    padded_feature_tensor = pad_sequence(feature_tensor, batch_first=True, padding_value=-1)
    
    # Stack targets - assuming targets are already all the same length or are scalars
    targets = torch.stack(targets)
    
    target_bin = torch.stack(target_bin)
    
    masks = torch.stack(masks)
    
    lengths = torch.tensor([len(seq) for seq in padded_feature_tensor])
    
    return padded_feature_tensor, targets, target_bin, masks, lengths

class EdgeModel(nn.Module):
    def __init__(self, hidden_size, num_edge_features, num_layers, num_edges, dropout, model_type):
        super().__init__()
        if model_type == 'rnn':
            self.rnn = nn.RNN(input_size=num_edge_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=num_edge_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size=num_edge_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, num_edges+1)

    def forward(self, x, hidden, mask, lengths):
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
    def __init__(self, wandb_config, model_config, data_config, train_config, test_config, model):
        super(Train_Model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_file_path = data_config['train_data_path']
        self.val_file_path = data_config['val_data_path']
        self.edge_features = data_config['edge_features']
        self.history_len = data_config['history_len']
        self.future_len = data_config['future_len']
        
        self.model_type = model_config['model_type']
        self.hidden_size = model_config['hidden_size']
        self.num_layers = model_config['num_layers']
        self.dropout = model_config['dropout']
        
        self.num_epochs = train_config['num_epochs']
        self.lr = train_config['lr']
        self.learning_rate_warmup_steps = train_config['learning_rate_warmup_steps']
        self.train_batch_size = train_config['batch_size']
        self.log_loss_every_steps = train_config['log_loss_every_steps']
        self.log_metrics_every_steps = train_config['log_metrics_every_steps']
        self.save = train_config['save_model']
        self.save_model_every_steps = train_config['save_model_every_steps']
        
        self.test_batch_size = test_config['batch_size']
        self.eval_every_steps = test_config['eval_every_steps']
        self.max_length = test_config['max_length']
        
        
        # WandB
        self.wandb_config = wandb_config
        wandb.init(
            settings=wandb.Settings(start_method="fork"),
            project=self.wandb_config['project'],
            entity=self.wandb_config['entity'],
            notes=self.wandb_config['notes'],
            job_type=self.wandb_config['job_type'],
            config={**data_config, **model_config, **train_config, **test_config}
        )
        self.exp_name = self.wandb_config['exp_name']
        wandb.run.name = self.wandb_config['run_name']
        
        # Logging
        self.model_dir = os.path.join("experiments", self.exp_name)

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
                # print("Features", features)
                # Sequence, target, mask = (bs=1, num_edges)
                batch_size = features.size(0)
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
                probabilities, preds = self.model(features, hidden, mask, lengths)
                loss = F.cross_entropy(probabilities, target)
                total_loss += loss.item()
                acc += (preds == target).sum().item() / batch_size
                loss.backward()
                self.optimizer.step()
                
            if self.save and (epoch + 1) % self.save_model_every_steps == 0:
                self.save_model()    
            
            if (epoch + 1) % self.log_loss_every_steps == 0:
                print("Epoch:", epoch)
                print("Avg Loss:", round(total_loss / len(self.train_dataloader), 5))
                wandb.log({"Epoch": epoch, "Average Train Loss": total_loss / len(self.train_dataloader)})
                wandb.log({"Epoch": epoch, "Train Accuracy": acc / len(self.train_dataloader)})
                print("Accuracy:", round(acc / len(self.train_dataloader), 5))
                wandb.log({"Epoch": epoch, "Random Baseline": 1 / self.train_dataset.avg_degree})

    def eval(self):
        sequences = []
        targets = []
        preds_list = []
        acc = 0
        with torch.no_grad():
            for features, target, target_bin, mask, lengths in self.val_dataloader:
                print("History", features[:, :, 0])
                print("Target", target)
                batch_size = features.size(0)
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
                current_input = features
                preds = []
                for step in range(self.max_length):
                    _, pred = self.model(current_input, hidden, mask, lengths+step)
                    print("Pred", pred)
                    if pred == self.stop_token or pred in preds:
                        break
                    preds.append(pred)
                    mask = torch.zeros((batch_size, self.num_edges+1), device=self.device)
                    for i in range(batch_size):
                        last_edge = self.val_dataset.edge_list[pred[i]]
                        # Find edges that share a node with the last edge
                        neighbors = [idx for idx, edge in enumerate(self.val_dataset.edge_list)
                                    if (edge[0] == last_edge[0] or edge[1] == last_edge[1]
                                        or edge[0] == last_edge[1] or edge[1] == last_edge[0])]
                        mask[i, neighbors] = 1
                        mask[i, pred] = 0
                    if torch.sum(mask[0]) == 0:
                        break
                    print("Mask", torch.argwhere(mask[0] == 1))
                    pred_input = self.get_edge_data(pred)
                    new_input = torch.zeros((batch_size, current_input.size(1) + 1, current_input.size(2)), device=self.device)
                    for i in range(batch_size):
                        new_input[i] = torch.cat([current_input[i], torch.cat([pred[i].unsqueeze(0), pred_input[i]]).unsqueeze(0)])
                    current_input = new_input
                preds_list.append(preds)
                    
            sequence = [torch.tensor([row[i, 0] for i in range(row.size(0)) if row[i, 0] != 0]) for row in features]
            sequences.append(sequence)
            targets.append(target)
            preds.append(pred_tensor)
            acc += (pred_tensor == target).sum().item() / batch_size
        avg_acc = acc / len(self.val_dataloader)
        wandb.log({"Val Accuracy": avg_acc})
        print("Val Accuracy:", avg_acc)
        return {"Sequences": sequences, "Targets": targets, "Predictions": preds}
    
    def get_edge_data(self, edge_indices):
        bs = edge_indices.size(0)
        edge_data = self.train_dataset.edge_coordinates[edge_indices].flatten()
        edge_data = edge_data.view(bs, -1)
        return edge_data
    
    def save_model(self):
        features = ''
        for feature in self.edge_features:
            features += feature + '_'
        save_path = os.path.join(self.model_dir, 
                                 self.exp_name + '_' + self.model_type + '_' + features + '_' + f'hist{self.history_len}' + f'_fut{self.future_len}' + f'_hidden_size_{self.hidden_size}.pth')
        torch.save(self.model.state_dict(), save_path)
        self.log.info(f"Model saved at {save_path}!")
        print(f"Model saved at {save_path}")
    
    def _build_train_dataloader(self):
        self.train_dataset = AlternativeAutoregressiveTrajectoryDataset(self.train_file_path, history_len=self.history_len, future_len=self.future_len, dataset='train', edge_features=self.edge_features, device=self.device)
        self.num_edge_features = 1
        if 'coordinates' in self.edge_features:
            self.num_edge_features += 4
        if 'edge_orientations' in self.edge_features:
            self.num_edge_features += 1
        if 'road_type' in self.edge_features:
            self.num_edge_features += self.train_dataset.num_road_types
        if 'pw_distance' in self.edge_features:
            self.num_edge_features += 1
        if 'edge_length' in self.edge_features:
            self.num_edge_features += 1
        if 'edge_angles' in self.edge_features:
            self.num_edge_features += 1
        self.num_edges = self.train_dataset.num_edges
        self.stop_token = self.num_edges
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, collate_fn=collate_fn)
        
    def _build_val_dataloader(self):
        self.val_dataset = AlternativeAutoregressiveTrajectoryDataset(self.val_file_path, history_len=self.history_len, future_len=self.future_len, dataset='val', edge_features=self.edge_features, device=self.device)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, collate_fn=collate_fn)

    def _build_model(self):
        self.model = self.model(self.hidden_size, self.num_edge_features, self.num_layers, self.num_edges, self.dropout, self.model_type)
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
        

data_config = {"dataset": "synthetic_20_traj",
    "train_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_20_traj.h5',
    "val_data_path": '/ceph/hdd/students/schmitj/MA_Diffusion_based_trajectory_prediction/data/synthetic_20_traj.h5',
    "history_len": 5,
    "future_len": 2,
    "num_classes": 2,
    "edge_features": ['coordinates'] # , 'road_type'
    }

model_config = {"model_type": "rnn", # rnn, lstm, or gru
    "hidden_size": 64,
    "num_layers": 3,
    "theta": 1.0, # controls strength of conv layers in residual model
    "dropout": 0.1,
    }

train_config = {"batch_size": 10,
    "optimizer": "adam",
    "lr": 0.0009,
    "num_epochs": 100,
    "learning_rate_warmup_steps": 80, # previously 10000
    "lr_decay": 0.999, # previously 0.9999
    "log_loss_every_steps": 1,
    "log_metrics_every_steps": 2,
    "save_model": False,
    "save_model_every_steps": 5}

test_config = {"batch_size": 1,
               "max_length": 30,
    "model_path": '/ceph/hdd/students/schmitj/experiments/synthetic_d3pm_test/synthetic_d3pm_test_edge_encoder_residual__hist5_fut5_marginal_prior_cosine_hidden_dim_64_time_dim_16_condition_dim_32.pth',
    "number_samples": 4,
    "eval_every_steps": 60
  }


wandb_config = {"exp_name": "benchmark_test",
                "run_name": "tdrive_benchmark_test",
                "project": "trajectory_prediction_using_denoising_diffusion_models",
                "entity": "joeschmit99",
                "job_type": "test",
                "notes": "",
                "tags": ["synthetic", "benchmark_rnn"]} 

model = Train_Model(wandb_config, model_config, data_config, train_config, test_config, model=EdgeModel)
model.train()
eval_dict = model.eval()
print("History sequence", eval_dict['Sequences'])
print("Target", eval_dict['Targets'])
print("Pred", eval_dict['Predictions'])

# hyperparameters of CSSRNN:
# hidden_size=400, fc size=200, train, val, test split of 80, 10, 10, LSTM layer, dropout of 0.1, gradient clipping at 1.0, rmsprop, lr=1e-4, lr_decay=0.9