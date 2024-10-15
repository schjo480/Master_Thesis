import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class EdgeRNN(nn.Module):
    def __init__(self, num_edge_features_rnn, num_edge_features, hidden_size, output_size, num_layers, dropout, model_type='rnn'):
        super(EdgeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size=num_edge_features_rnn, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.feature_encoding_1 = nn.Linear(num_edge_features, self.hidden_size)
        self.feature_encoding_2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x, hidden, masks, feature_tensor=None):
        feature_encoding = F.relu(self.feature_encoding_1(feature_tensor))  # (batch_size, seq_length, num_edges, hidden_size)
        feature_encoding = self.feature_encoding_2(feature_encoding).squeeze(-1)   # (batch_size, seq_length, num_edges)
        
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # (batch_size, seq_length, num_edges)
        out = out + feature_encoding

        out = out.masked_fill(~masks, float(-10000))
        # Apply softmax to get probabilities
        out = F.log_softmax(out, dim=-1)

        return out, hidden


    def init_hidden(self, batch_size, device):
        # Initialize hidden state
        if self.model_type == 'lstm':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            return (h0, c0)
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
