# -*- coding: utf-8 -*-
import json, os
import math

import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F


# bilstm的gcn
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(in_features, out_features)

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))  
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features)) 
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) 
        self.weight.data.uniform_(-stdv, stdv)  
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, text, adj):
        
        batch_size, N, _ = text.shape
        I = torch.eye(N, device=adj.device).unsqueeze(0)  
        A_hat = adj + I  

        D_hat = torch.sum(A_hat, dim=2, keepdim=True)  
        D_hat_sqrt_inv = 1.0 / torch.sqrt(D_hat)    

        D_hat_sqrt_inv = D_hat_sqrt_inv.expand(batch_size, N, N)  
        A_norm = D_hat_sqrt_inv * A_hat * D_hat_sqrt_inv  

        hidden = torch.matmul(text, self.weight)
        H = torch.bmm(A_norm, hidden) 
        
        H = self.linear(H)  

        if self.bias is not None:
            return H + self.bias
        else:
            return H

class GCNModel(nn.Module):
    def __init__(self, args):
        super(GCNModel, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)
        self.gc2 = GraphConvolution(2*args.lstm_dim, 2*args.lstm_dim)

    def forward(self, lstm_feature, sentence_adjs , mask):
        inputs = lstm_feature
        adjs = sentence_adjs
        x = F.relu(self.gc1(inputs, adjs))
        x = F.relu(self.gc2(x, adjs))
        x = x * mask.unsqueeze(2).float().expand_as(x)

        output = x
        return output