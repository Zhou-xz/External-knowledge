import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


class InteractionSelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(InteractionSelfAttention, self).__init__()
        self.args = args
        self.q = nn.Linear(args.hidden_size, args.hidden_size)
        self.k = nn.Linear(args.hidden_size, args.hidden_size)
        self.v = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, args.hidden_size)

        self.q_1 = nn.Linear(2*args.lstm_dim, 2*args.lstm_dim)
        self.k_1 = nn.Linear(2*args.lstm_dim, 2*args.lstm_dim)
        self.v_1 = nn.Linear(2*args.lstm_dim, 2*args.lstm_dim)
        self.fc_1 = nn.Linear(2*args.lstm_dim, 2*args.lstm_dim)


    def forward(self, query, value, mask):
        batch_size, N, dim = query.shape
        if dim==768:
            attention_states = self.q(query)
            attention_states_T = self.k(value)
            attention_states_T = attention_states_T.permute([0, 2, 1])  
            value = self.v(value)
        else:
            attention_states = self.q_1(query)
            attention_states_T = self.k_1(value)
            attention_states_T = attention_states_T.permute([0, 2, 1])  
            value = self.v_1(value)

        weights = torch.bmm(attention_states, attention_states_T) 
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights, dim=2)

        merged = torch.bmm(attention, value)
        merged = merged * mask.unsqueeze(2).float().expand_as(merged)
        if dim==768:
            merged = self.fc(merged)
        else:
            merged = self.fc_1(merged)

        return merged

