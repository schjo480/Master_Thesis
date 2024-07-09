import torch
import torch.nn.functional as F
import torch.nn as nn

class Benchmark_MLP(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, num_edges, hidden_channels, num_edge_features):
        super(Benchmark_MLP, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.num_classes = num_classes        
        
        self.hidden_channels = hidden_channels
        self.num_layers = self.config['num_layers']
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(self.num_edge_features, self.hidden_channels))
        for _ in range(1, self.num_layers):
            self.lin_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        
    def forward(self, history_edge_attributes, last_history_edge, line_graph):
        x = history_edge_attributes
        for layer in self.lin_layers:
            x = F.relu(layer(x))
        
        logits_list = []
        preds = []
        
        for _ in range(self.future_len):
            # Get neighbors of the last edge in the trajectory
            neighbors = list(line_graph.neighbors(last_history_edge.item()))
            neighbor_features = x[:, neighbors, :]
            
            # Calculate probability for next edge
            logits = self.output_layer(neighbor_features)
            logits_list.append(logits)
            
            # Update last_history_edge with the most probable edge
            _, most_probable_edge_idx = torch.max(logits, dim=-1)
            preds.append(most_probable_edge_idx)
            print("Preds", preds)
            last_history_edge = neighbors[most_probable_edge_idx.item()]
        
        preds = torch.one_hot(torch.cat(preds, dim=0), num_classes=self.num_edges)
        logits = torch.cat(logits_list, dim=1)
        return logits, preds