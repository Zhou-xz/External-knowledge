import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F
import numpy


class CrossAttention(torch.nn.Module):
    def __init__(self, args):
        super(CrossAttention, self).__init__()

        self.args = args
        self.hidden_dim = args.hidden_size 
        self.lstm_dim = args.lstm_dim  

        self.syntax_to_semantic = CrossAttentionLayer(self.hidden_dim, self.lstm_dim * 2, self.hidden_dim)
        self.semantic_to_syntax = CrossAttentionLayer(self.lstm_dim * 2, self.hidden_dim, self.hidden_dim)
        self.gate = nn.Linear(self.hidden_dim * 2, 2)  

    def forward(self, syntactic_feat, semantic_feat):
        enhanced_semantic = self.syntax_to_semantic(semantic_feat, syntactic_feat)  
        enhanced_syntax = self.semantic_to_syntax(syntactic_feat, semantic_feat)    
        
        return enhanced_semantic, enhanced_syntax

class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, key_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, query_feat, key_value_feat):
        Q = self.query_proj(query_feat)          
        K = self.key_proj(key_value_feat)        
        V = self.value_proj(key_value_feat)      
        
        attn = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)          
        
        return output
