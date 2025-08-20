import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from .table import TableEncoder
from .matching_layer import MatchingLayer
from .multihead_attention import MultiHeadSelfAttention
from .char_cnn import CharCNN
from .fusion_layer import FusionLayer
from .boundary_contrastive import ContrastiveLoss, compute_batch_boundary_contrastive_loss


class BDTFModel(BertPreTrainedModel):
    def __init__(self, config, gen_embed=None, domain_embed=None, char_vocab_size=None):
        super().__init__(config)

        self.bert = BertModel(config)
        self.char_cnn = CharCNN(char_vocab_size)
        if config.use_interaction: 
            self.fusion = FusionLayer(config, gen_embed, domain_embed)
        self.table_encoder = TableEncoder(config)
        self.inference = InferenceLayer(config)
        self.matching = MatchingLayer(config)
        self.mha = MultiHeadSelfAttention(config)
        self.MLP1 = torch.nn.Linear(768, 300)
        self.MLP2 = torch.nn.Linear(768, 300)
        self.triplet_biaffine_S = Biaffine(config, 300, 300, 1, bias=(True, True))
        self.triplet_biaffine_E = Biaffine(config, 300, 300, 1, bias=(True, True))
    
        self.init_weights()

    def forward(self, input_ids, char_indices, bert_attention_mask, attention_mask, lstm_mask, ids,
                start_label_masks, end_label_masks,
                t_start_labels=None, t_end_labels=None,
                o_start_labels=None, o_end_labels=None,
                table_labels_S=None, table_labels_E=None,
                polarity_labels=None, pairs_true=None,
                bert_token_position=None, lstm_tokens=None,
                pos_packs=None, adj_packs=None):
        seq = self.bert(input_ids, bert_attention_mask)[0]  
        seq = self.align(seq, bert_token_position)

        char_embeds = self.char_cnn(char_indices)

        out, attention_weights = self.mha(seq)
        attention_weights = torch.mean(attention_weights, dim = 1) 

        if self.config.use_interaction:  
            length = lstm_mask.sum(1) 
            seq = self.fusion(seq, char_embeds, attention_weights, lstm_tokens, pos_packs, adj_packs, length, lstm_mask)

        ap_node_S = F.relu(self.MLP1(seq))  
        op_node_S = F.relu(self.MLP2(seq))  

        biaffine_edge_S = self.triplet_biaffine_S(ap_node_S, op_node_S) 
        biaffine_edge_E = self.triplet_biaffine_E(ap_node_S, op_node_S)

        biaffine_edge_S = torch.sigmoid(biaffine_edge_S) 
        biaffine_edge_E = torch.sigmoid(biaffine_edge_E) 


        table = self.table_encoder(seq, attention_mask)

        loss_fn = ContrastiveLoss(margin=1.0)
        boundary_loss = compute_batch_boundary_contrastive_loss(table, pairs_true, loss_fn)
        output = self.inference(table, attention_mask, table_labels_S, table_labels_E, biaffine_edge_S, biaffine_edge_E)
        output['ids'] = ids
        output['cl_loss'] = boundary_loss
        output = self.matching(output, table, pairs_true, seq)
        return output

    def align(self, seq, position):  
        bert_state = torch.zeros(position.size(0), position.size(1), seq.size(2))  
        for i in range(position.size(0)):
            for j, index in enumerate(position[i].tolist()):
                bert_state[i][j] = torch.sum(seq[i][index[0]:index[1]], dim=0)

        return bert_state.to(seq.device)


class InferenceLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_linear_S = nn.Linear(768, 1)
        self.cls_linear_E = nn.Linear(768, 1)

    def span_pruning(self, pred, z, attention_mask):
        mask_length = attention_mask.sum(dim=1) - 2  
        length = ((attention_mask.sum(dim=1) - 2) * z).long() 
        length[length < 5] = 5 
        max_length = mask_length ** 2  
        for i in range(length.shape[0]):
            if length[i] > max_length[i]:
                length[i] = max_length[i]
        batch_size = attention_mask.shape[0] 
        pred_sort, _ = pred.view(batch_size, -1).sort(descending=True) 
        batchs = torch.arange(batch_size).to('cuda')
        
        topkth = pred_sort[batchs, length - 1].unsqueeze(1) 
        return pred >= (topkth.view(batch_size, 1, 1)) 
 
    def forward(self, table, attention_mask, table_labels_S, table_labels_E, biaffine_edge_S, biaffine_edge_E):
        outputs = {}

        logits_S_temp = torch.squeeze(self.cls_linear_S(table), 3)  
        logits_E_temp = torch.squeeze(self.cls_linear_E(table), 3)  

        logits_S = logits_S_temp * (1 + torch.squeeze(biaffine_edge_S, 3))
        logits_E = logits_E_temp * (1 + torch.squeeze(biaffine_edge_E, 3))

        loss_func = nn.BCEWithLogitsLoss(weight=(table_labels_S >= 0))

        outputs['table_loss_S'] = loss_func(logits_S, table_labels_S.float()) 
        outputs['table_loss_E'] = loss_func(logits_E, table_labels_E.float()) 
        
        
        S_pred = torch.sigmoid(logits_S) * (table_labels_S >= 0)
        E_pred = torch.sigmoid(logits_E) * (table_labels_S >= 0)

        if self.config.span_pruning != 0:  
            table_predict_S = self.span_pruning(S_pred, self.config.span_pruning, attention_mask)
            table_predict_E = self.span_pruning(E_pred, self.config.span_pruning, attention_mask)
        else:
            table_predict_S = S_pred > 0.5
            table_predict_E = E_pred > 0.5
        outputs['table_predict_S'] = table_predict_S
        outputs['table_predict_E'] = table_predict_E
        outputs['table_labels_S'] = table_labels_S
        outputs['table_labels_E'] = table_labels_E
        return outputs



class Biaffine(torch.nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.bia_linear = torch.nn.Linear(in_features=self.linear_input_size,
                                          out_features=self.linear_output_size,
                                          bias=False) 

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).cuda()
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).cuda()
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.bia_linear(input1)
        affine = affine.view(batch_size, len1 * self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine
