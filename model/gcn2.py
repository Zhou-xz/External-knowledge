# -*- coding: utf-8 -*-
import json, os
import math

import numpy
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

        self.in_features = in_features
        self.out_features = out_features
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
        batch_size, N, _ = adj.shape
        adj = adj + torch.eye(N, device=adj.device).unsqueeze(0) 
        
        degree = torch.sum(adj, dim=2, keepdim=True)  
        degree_inv = torch.where(degree > 0, 1.0 / degree, 0.0)  
        norm_adj = degree_inv * adj  

        hidden = torch.matmul(text, self.weight)
        output = torch.bmm(norm_adj, hidden)      

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCNModel2(nn.Module):
    def __init__(self, args):
        super(GCNModel2, self).__init__()
        self.args = args
        self.gc1 = GraphConvolution(args.hidden_size, args.hidden_size)
        self.gc2 = GraphConvolution(args.hidden_size, args.hidden_size)

    def forward(self, lstm_feature, sentence_adjs , mask):
        inputs = lstm_feature
        adjs = sentence_adjs
        x = F.relu(self.gc1(inputs, adjs))
        x = F.relu(self.gc2(x, adjs))
        x = x * mask.unsqueeze(2).float().expand_as(x)

        output = x
        return output
