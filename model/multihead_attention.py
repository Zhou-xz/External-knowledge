import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadSelfAttention, self).__init__()
        self.args = args
        self.embed_dim = args.hidden_size          
        self.num_heads = args.num_attention_heads 
        self.head_dim = args.hidden_size // args.num_attention_heads 
        assert self.embed_dim % self.num_heads == 0

        self.query = nn.Linear(self.embed_dim, self.embed_dim)
        self.key = nn.Linear(self.embed_dim, self.embed_dim)
        self.value = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.relative_position_bias = nn.Parameter(
            torch.randn(2 * args.max_position_embeddings - 1, self.num_heads) * 0.02
        )  
        
        self.register_buffer(
            "relative_position_index",
            self._generate_relative_positions_matrix(args.max_position_embeddings, args.max_position_embeddings)
        )

    @staticmethod
    def _generate_relative_positions_matrix(query_len, key_len):
        
        q_ids = torch.arange(query_len)[:, None]  
        k_ids = torch.arange(key_len)[None, :]    
        return (q_ids - k_ids) + (key_len - 1)    

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        
        Q = self.query(x)  # [batch, seq_len, embed_dim]
        K = self.key(x)     # [batch, seq_len, embed_dim]
        V = self.value(x)   # [batch, seq_len, embed_dim]
        
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        content_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        
        relative_pos_bias = self.relative_position_bias[
            self.relative_position_index[:seq_length, :seq_length]
        ].permute(2, 0, 1)  # [heads, seq_len, seq_len]
        
        attention_scores = (content_scores + relative_pos_bias.unsqueeze(0)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        out = torch.matmul(attention_weights, V)  # [batch, heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        out = self.fc_out(out)
        
        return out, attention_weights