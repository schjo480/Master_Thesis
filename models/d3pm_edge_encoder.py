import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATv2Conv


def get_timestep_embedding(timesteps, embedding_dim, max_time=1000.):
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
    assert timesteps.dim() == 1  # Ensure timesteps is a 1D tensor

    # Scale timesteps by the maximum time
    timesteps = timesteps.float() * (1000. / max_time)

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # Add zero-padding if embedding dimension is odd
        zero_pad = torch.zeros((timesteps.shape[0], 1), dtype=torch.float32)
        emb = torch.cat([emb, zero_pad], dim=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class Edge_Encoder(nn.Module):
    def __init__(self, model_config, history_len, future_len, num_classes, nodes, edges, node_features, num_edges, hidden_channels, num_edge_features):
        super(Edge_Encoder, self).__init__()
        # Config
        self.config = model_config
        
        # Data
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.edges = edges
        self.node_features = node_features
        self.num_node_features = self.node_features.shape[1]
        self.num_edges = num_edges
        self.num_edge_features = num_edge_features
        self.history_len = history_len
        self.future_len = future_len
        
        
        self.num_classes = num_classes
        # self.out_channels = self.config['model']['args']['out_channels']
        self.model_output = self.config['model_output']
        
        # Time embedding
        self.max_time = 1000.
        self.time_embedding_dim = self.config['time_embedding_dim']
        self.time_linear0 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        self.time_linear1 = nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
    
        # TODO: Adapt model dimensions to fit the new input data
        # New history data is of shape (bs, num_edges) and not (bs, history_len)
        # New future data is of shape (bs, num_edges) and not (bs, future_len)
        # Model
        # GNN layers
        self.hidden_channels = hidden_channels
        self.num_heads = self.config['num_heads']
        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads)
        self.conv2 = GATv2Conv(self.hidden_channels*self.num_heads, self.hidden_channels, edge_dim=self.num_edge_features, heads=self.num_heads)
        
        # Output layers for each task
        self.condition_dim = self.config['condition_dim']
        self.history_encoder = nn.Linear(self.hidden_channels*self.num_heads, self.condition_dim)  # To encode history to c
        self.future_decoder = nn.Linear(self.hidden_channels, self.num_edges)  # To predict future edges
        self.adjust_to_future_len = nn.Conv1d(in_channels=self.num_nodes, out_channels=self.future_len, kernel_size=1)

    def forward(self, x, edge_index, t=None, edge_attr=None, condition=None, mode=None):
        '''
        Forward pass through the model
        Args:
            x: torch.Tensor: input tensor: noised future trajectory indices / history trajectory indices
            t: torch.Tensor: timestep tensor'''    
        
        # GNN forward pass
        # TODO: Integrate edge indices of trajectories (possibly just do it with mask as edge attribute?)
        # --> This works if we are working with a batch size of 1
        # --> Possibly need gradient accumulation for faster training
        
        # Edge Embedding
        x = F.relu(self.conv1(x, edge_index, edge_attr.squeeze(0)))
        x = F.relu(self.conv2(x, edge_index, edge_attr.squeeze(0)))
        x = x.unsqueeze(0).repeat(edge_attr.size(0), 1, 1) # Reshape x to [batch_size, num_nodes, feature_size]
        # x = torch.cat((x, edge_emb), dim=1)
        
        
        if mode == 'history':
            c = self.history_encoder(x)
            
            return c
        
        elif mode == 'future':
            # Time embedding
            t_emb = get_timestep_embedding(t, embedding_dim=self.time_embedding_dim, max_time=self.max_time)
            t_emb = self.time_linear0(t_emb)
            t_emb = F.silu(t_emb)  # SiLU activation, equivalent to Swish
            t_emb = self.time_linear1(t_emb)
            t_emb = F.silu(t_emb)
            t_emb = t_emb.unsqueeze(1).repeat(1, self.num_nodes, 1)
            #Concatenation
            x = torch.cat((x, t_emb), dim=2) # Concatenate with time embedding
            x = torch.cat((x, condition), dim=2) # Concatenate with condition c
            x = F.relu(nn.Linear(x.size(2), self.hidden_channels)(x))
            
            logits = self.future_decoder(x)
            logits = F.relu(self.adjust_to_future_len(logits))
            # TODO: Mimicking original code: (bs, height, width, self.out_ch, self.num_pixel_vals)
            # this model should output logits of the shape (bs, future_len, 1, num_edges)
            # Currently it outputs (bs, future_len, num_edges)
            logits = logits.unsqueeze(2) # (bs, future_len, num_edges) -> (bs, future_len, 1, num_edges)

            return logits            