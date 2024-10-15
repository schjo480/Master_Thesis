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