import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from .interaction_attention import InteractionSelfAttention
from .cross_attention import CrossAttention
from .gcn import GCNModel
from .gcn2 import GCNModel2

class FusionLayer(nn.Module):

    def __init__(self, config, gen_emb, domain_emb):
        super().__init__()

        self.config = config
        self.hidden_dim = config.hidden_size
        self.lstm_dim = config.lstm_dim
        self.final = nn.Linear(config.lstm_dim * 2, config.hidden_size)
        self.loop = nn.Linear(config.hidden_size, config.lstm_dim * 2)

        self.fc1 = nn.Linear(config.hidden_size, 1)
        self.fc2 = nn.Linear(config.hidden_size, 1)
        self.fc3 = nn.Linear(config.hidden_size, 1)
        self.fc4 = nn.Linear(config.hidden_size, 1)

        self.general_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1]) 
        self.general_embedding.weight.data.copy_(gen_emb) 
        self.general_embedding.weight.requires_grad = False 

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])  
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False

        self.pos_embedding = torch.nn.Embedding(5, config.pos_dim) 

        
        self.dropout1 = torch.nn.Dropout(0.1) 
        self.dropout2 = torch.nn.Dropout(0.25)

        # Bilstm encoder 
        self.bilstm_encoder = torch.nn.LSTM(300 + 500 + config.pos_dim + 100, config.lstm_dim, num_layers=1, batch_first=True,
                                            bidirectional=True)
       
        self.bilstm = torch.nn.LSTM(config.lstm_dim * 2, config.lstm_dim, num_layers=1, batch_first=True,
                                    bidirectional=True)

        # GCN Module
        self.gcn_layer = GCNModel(config)
        self.gcn_layer2 = GCNModel2(config)
       
        self.interaction_attention = InteractionSelfAttention(config)
        self.cross_attention = CrossAttention(config)


    def _get_double_embedding(self, sentence_tokens, mask):
        general_embedding = self.general_embedding(sentence_tokens)
        domain_embedding = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([general_embedding, domain_embedding], dim=2)  
        embedding = self.dropout1(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)  
        return embedding

    def _get_pos_embedding(self, sentence_poses, mask):
        pos_embedding = self.pos_embedding(sentence_poses)
        embedding = pos_embedding * mask.unsqueeze(2).float().expand_as(pos_embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)  
        context, _ = self.bilstm_encoder(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)  
        return context

    def _lstm_feature1(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True, enforce_sorted=False)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def forward(self, bert_feature1, char_embeds, bert_feature, sentence_tokens, pos_pack, adj_pack, lengths, masks):
        origin_bert_feature = bert_feature1
        mask = torch.nn.functional.pad(masks, [1, 1]) 
        double_embedding = self._get_double_embedding(sentence_tokens, masks)
        pos_embedding = self._get_pos_embedding(pos_pack, masks)
        cnn_embedding = char_embeds * masks.unsqueeze(2).float().expand_as(char_embeds)
    
        embedding = torch.cat([double_embedding, pos_embedding, cnn_embedding], dim=2)
        # Bilstm encoder
        lstm_feature = self._lstm_feature(embedding, lengths) 
        # GCN encoder
        lstm_feature = torch.nn.functional.pad(lstm_feature, [0, 0, 1, 1])
        adj = torch.nn.functional.pad(adj_pack, [1, 1, 1, 1])
        # adj = adj_pack
        gcn_feature = self.gcn_layer(lstm_feature, adj, mask) 
        bert_feature = self.gcn_layer2(bert_feature1, bert_feature, mask)
        
        a = bert_feature  
        b = lstm_feature  
        c = self.final(b)


        gcn_interaction_attention = self.interaction_attention(gcn_feature, gcn_feature, mask)
        bert_interaction_attention = self.interaction_attention(bert_feature, bert_feature, mask)

        bert_interaction_feature_drop = self.dropout2(bert_interaction_attention) + a
        gcn_interaction_feature_drop = self.dropout2(gcn_interaction_attention)  + b

        enhanced_semantic, enhanced_syntax = self.cross_attention(gcn_interaction_feature_drop, bert_interaction_feature_drop)

        gcn_feature = enhanced_syntax
        bert_feature = enhanced_semantic
        g1 = self.fc1(enhanced_semantic)  
        g2 = self.fc2(enhanced_syntax)  
        g3 = self.fc3(origin_bert_feature)  
        g4 = self.fc4(c)  

        scores = torch.cat([g1, g2, g3, g4], dim=-1) 
        gates = F.softmax(scores, dim=-1)  
        stacked = torch.stack([enhanced_semantic, enhanced_syntax, origin_bert_feature, c], dim=-2) 
        fused = (gates.unsqueeze(-1) * stacked).sum(dim=-2) 
    
        return fused
