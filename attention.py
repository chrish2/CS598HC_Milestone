from unicodedata import bidirectional
import torch.nn as nn
import torch
import torch.nn.functional as F


class BiLSTM_Attention(nn.Module):
    def __init__(self, batch_size, hidden_dim, lstm_layers, max_words):
        super(BiLSTM_Attention, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = max_words  
        self.dropout = nn.Dropout(0.5)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=self.LSTM_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(in_features=self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.hidden_dim, 1)  
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, x):
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        out = self.embedding(x)
        out, (final_hidden_state, _) = self.lstm(out, (h, c))
        out = self.attention_net(out, final_hidden_state)
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out))
        out = self.dropout(out)
        return torch.sigmoid(self.fc2(out))