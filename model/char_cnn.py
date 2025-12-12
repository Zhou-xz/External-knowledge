import torch
import torch.nn as nn
import torch.nn.functional as F



class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, embed_dim=16, num_filters=100, filter_size=5):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, filter_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        batch_size, seq_len, char_len = x.shape
        x = x.view(-1, char_len)  
        embedded = self.embedding(x).permute(0, 2, 1)  
        conv_out = self.conv(embedded)  
        pooled = self.pool(conv_out).squeeze(-1)  
        return pooled.view(batch_size, seq_len, -1) 