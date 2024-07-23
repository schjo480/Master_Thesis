import torch
import torch.nn as nn
import torch.nn.functional as F


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

def generate_positional_encodings(indices, history_len, embedding_dim, device):
    """
    Generates positional encodings only for specified indices.

    Args:
        indices (torch.Tensor): The indices of history edges.
        history_len (int): Total positions (here, history length).
        embedding_dim (int): The dimensionality of each position encoding.
        device (torch.device): The device tensors are stored on.

    Returns:
        torch.Tensor: Positional encodings for specified indices with shape [len(indices), embedding_dim].
    """
    positions = torch.arange(history_len, dtype=torch.float32, device=device)
    # Get div term
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
    div_term = div_term.to(device)

    # Compute positional encodings
    pe = torch.zeros((len(indices), embedding_dim), device=device)
    pe[:, 0::2] = torch.sin(positions[indices][:, None] * div_term[None, :])
    pe[:, 1::2] = torch.cos(positions[indices][:, None] * div_term[None, :])

    return pe

class Edge_Encoder_MLP(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, num_edges, hidden_channels, edge_features, num_edge_features, num_timesteps, pos_encoding_dim):
        super(Edge_Encoder_MLP, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.num_edges = num_edges
        self.edge_features = edge_features
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        self.num_classes = num_classes
        
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
        self.num_layers = self.config['num_layers']
        self.lin_layers = nn.ModuleList()
        self.lin_layers.append(nn.Linear(self.num_edge_features, self.hidden_channels))
        for _ in range(1, self.num_layers):
            self.lin_layers.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels, self.condition_dim)  # To encode history to c
        if 'pos_encoding' in self.edge_features:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim + self.pos_encoding_dim,
                                        self.hidden_channels)  # To predict future edges
        else:
            self.future_decoder = nn.Linear(self.hidden_channels + self.condition_dim + self.time_embedding_dim,
                                            self.hidden_channels)  # To predict future edges
        self.adjust_to_class_shape = nn.Linear(self.hidden_channels, self.num_classes)

    def forward(self, x, indices=None, t=None, condition=None, mode=None):
        """
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor
        """    
        # GNN forward pass
        
        # Edge Embedding        
        for layer in self.lin_layers:
            x = F.relu(layer(x))

        if mode == 'history':
            c = self.history_encoder(x) # (bs, num_edges, condition_dim)
            if 'pos_encoding' in self.edge_features:
                pos_encoding = generate_positional_encodings(indices, self.history_len, self.pos_encoding_dim, device=x.device)
                encodings = F.silu(self.pos_linear0(pos_encoding))
                encodings = F.silu(self.pos_linear1(encodings))
                c = self.integrate_encodings(c, indices, encodings)
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time, device=x.device)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)   # (bs, time_embedding_dim)
            t_emb = t_emb.unsqueeze(1).repeat(1, x.size(1), 1) # (bs, num_edges, time_embedding_dim)
            
            #Concatenation
            x = torch.cat((x, t_emb), dim=2) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=2) # Concatenate with condition c, (bs, num_edges, hidden_channels + condition_dim + time_embedding_dim)
            
            logits = self.future_decoder(x) # (bs, num_edges, hidden_channels)
            logits = self.adjust_to_class_shape(logits) # (bs, num_edges, num_classes=2)

            return logits
        
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
        features = features.to(encodings.device)
        
        # Expand the features tensor to accommodate the positional encodings
        new_features = torch.cat([features, torch.zeros(features.size(0), encodings.size(1), device=features.device)], dim=1)
        
        # Place the positional encodings in the rows specified by indices
        new_features[indices, -encodings.size(1):] = encodings

        return new_features