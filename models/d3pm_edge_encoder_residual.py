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
    
def generate_positional_encodings(history_len, embedding_dim, device, n=1000):
    PE = torch.zeros((history_len, embedding_dim), device=device)
    for k in range(history_len):
        for i in torch.arange(int(embedding_dim/2)):
            denominator = torch.pow(n, 2*i/embedding_dim)
            PE[k, 2*i] = torch.sin(k/denominator)
            PE[k, 2*i+1] = torch.cos(k/denominator)
    return PE

class Edge_Encoder_Residual(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, edge_features, num_edge_features, num_timesteps, pos_encoding_dim):
        super(Edge_Encoder_Residual, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.edge_features = edge_features
        
        self.num_classes = num_classes
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = num_timesteps
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        
        # Positional Encoding
        self.pos_encoding_dim = pos_encoding_dim
        if 'pos_encoding' in self.edge_features:
            self.pos_linear0 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
            self.pos_linear1 = nn.Linear(self.pos_encoding_dim, self.pos_encoding_dim)
    
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.theta = self.config['theta']
        
        self.convs = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        if 'pos_encoding' in self.edge_features:
            self.convs.append(GATv2Conv(self.num_edge_features + self.pos_encoding_dim + self.time_embedding_dim, self.hidden_channels, heads=self.num_heads))
            self.res_layers.append(nn.Linear(self.num_edge_features + self.pos_encoding_dim + self.time_embedding_dim, self.hidden_channels * self.num_heads))
        else: 
            self.convs.append(GATv2Conv(self.num_edge_features + self.time_embedding_dim, self.hidden_channels, heads=self.num_heads))
            self.res_layers.append(nn.Linear(self.num_edge_features + self.time_embedding_dim, self.hidden_channels * self.num_heads))
            
        for _ in range(1, self.num_layers):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, bias=False))
            self.res_layers.append(nn.Linear(self.hidden_channels * self.num_heads, self.hidden_channels * self.num_heads))

        # Output layers for each task
        self.future_decoder = nn.Linear(self.hidden_channels * self.num_heads, self.hidden_channels)  # To predict future edges
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, edge_index, t, indices=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: edge attributes, including binary encoding fo edges present/absent in trajectory
            t: torch.Tensor: timestep tensor
        """    
        
        # GNN forward pass
        # Edge Embedding
        batch_size = x.size(0) // self.num_edges
        if 'pos_encoding' in self.edge_features:
            pos_encoding = generate_positional_encodings(self.history_len, self.pos_encoding_dim, device=x.device)
            encodings = F.silu(self.pos_linear0(pos_encoding))
            encodings = F.silu(self.pos_linear1(encodings))
            x = self.integrate_encodings(x, indices, encodings) # (batch_size * num_edges, num_edge_features + pos_encoding_dim)
        
        # Time embedding
        t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
        t_emb = self.time_linear0(t_emb)
        t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
        t_emb = self.time_linear1(t_emb)
        t_emb = F.silu(t_emb)   # (batch_size, time_embedding_dim)
        t_emb = torch.repeat_interleave(t_emb, self.num_edges, dim=0) # (batch_size * num_edges, time_embedding_dim)
        
        x = torch.cat((x, t_emb), dim=1) # Concatenate with time embedding, (batch_size * num_edges, num_edge_features + time_embedding_dim (+ pos_enc_dim))
        
        for conv, res_layer in zip(self.convs, self.res_layers):
            res = F.relu(res_layer(x))
            x = F.relu(conv(x, edge_index))
            x = self.theta * x + res        # (batch_size * num_edges, hidden_channels * num_heads)        

        logits = self.future_decoder(x) # (batch_size * num_edges, hidden_channels)
        logits = self.adjust_to_class_shape(logits) # (batch_size * num_edges, num_classes=2)

        return logits.view(batch_size, self.num_edges, -1)  # (batch_size, num_edges, num_classes=2)

    def integrate_encodings(self, features, indices, encodings):
        """
        Integrates positional encodings into the feature matrix.

        Args:
            features (torch.Tensor): Original feature matrix [num_edges, num_features].
            indices (torch.Tensor): Indices of edges where the encodings should be added.
            encodings (torch.Tensor): Positional encodings [len(indices), encoding_dim].

        Returns:
            torch.Tensor: Updated feature matrix with positional encodings integrated.
        """
        
        # Ensure that features are on the same device as encodings
        batch_size = indices.shape[0]
        encodings = encodings.repeat(batch_size, 1)
        # Ensure that features and encodings are on the same device
        features = features.to(encodings.device)
        
        # Expand the features tensor to accommodate the positional encodings
        new_features = torch.cat([features, torch.zeros(features.size(0), encodings.size(1), device=features.device)], dim=1)
        
        # Calculate batch offsets
        batch_offsets = torch.arange(batch_size, device=features.device) * self.num_edges
        
        # Flatten indices for direct access, adjust with batch offsets
        flat_indices = (indices + batch_offsets.unsqueeze(1)).flatten()
        
        # Place the positional encodings in the correct rows across all batches
        new_features[flat_indices, -encodings.size(1):] = encodings

        return new_features
