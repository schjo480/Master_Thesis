import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000., device=None):
    """
    Build sinusoidal embeddings (from Fairseq).

    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".

    Args:
        timesteps: torch.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        max_time: float: largest time input

    Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    timesteps = timesteps.to(device)
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb.to(device)

class Edge_Encoder_Residual(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, num_edge_features, num_timesteps):
        super(Edge_Encoder_Residual, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = num_timesteps
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.theta = self.config['theta']
        
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(self.num_edge_features, self.hidden_channels, heads=self.num_heads))
        self.res_layers = nn.ModuleList()
        self.res_layers.append(nn.Linear(self.num_edge_features, self.hidden_channels * self.num_heads))
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, bias=False))
            self.res_layers.append(nn.Linear(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads))

        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels * self.num_heads, self.condition_dim)  # To encode history to c
        self.future_decoder = nn.Linear(self.hidden_channels * self.num_heads + self.condition_dim + self.time_embedding_dim,
                                        self.hidden_channels)
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, edge_index, t=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: edge attributes, including binary encoding fo edges present/absent in trajectory
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        # Edge Embedding
        if x.dim() == 3:
            x = x.squeeze(0)    # (bs, num_edges, num_edge_features) -> (num_edges, num_edge_features)
        
        for conv, res_layer in zip(self.convs, self.res_layers):
            res = F.relu(res_layer(x))
            x = F.relu(conv(x, edge_index)) # (num_edges, hidden_channels)
            x = self.theta * x + res
            
        if mode == 'history':
            c = self.history_encoder(x)
            
            return c
        
        elif mode == 'future':
            # Time embedding
            # TODO: Check if time embedding and condition generation are the same for each datapoint in a batch and hence the problems while sampling
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)
            t_emb = t_emb.repeat(self.num_edges, 1) # (num_edges, time_embedding_dim)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=1) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=1) # Concatenate with condition c
            
            logits = self.future_decoder(x) # (num_edges, hidden_channels)
            logits = self.adjust_to_class_shape(logits) # (num_edges, num_classes=2)

            return logits.unsqueeze(0)  # (1, num_edges, num_classes=2)
