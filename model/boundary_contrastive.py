import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        

    def forward(self, anchor, positive, negative):
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return loss.mean()

def generate_negative_samples(table, x0, x1):
    seq_len = table.size(0)
    negatives = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  
    for dx, dy in directions:
        new_x0 = x0 + dx
        new_x1 = x1 + dy
        if 0 <= new_x0 < seq_len and 0 <= new_x1 < seq_len:
            negatives.append(table[new_x0, new_x1,:])

    if len(negatives) == 0:
        negatives.append(table[0, 0, 0])

    negatives = torch.stack(negatives, dim=0)
    return negatives

def compute_boundary_contrastive_loss(table, pairs, loss_fn, mode='S'):
    anchor_list = []
    positive_list = []
    negative_list = []

    for triplet in pairs:
        s0, e0, s1, e1, p = triplet

        if mode == 'S':
            x0_idx = s0 + 1
            x1_idx = s1 + 1
        elif mode == 'E':
            x0_idx = e0
            x1_idx = e1
        else:
            raise ValueError(f"Unsupported mode: {mode}, should be 'S' or 'E'.")

        x_vector = table[x0_idx, x1_idx,:]
        pos_vector = x_vector + 0.1 * torch.randn_like(x_vector)
        anchor_vector = x_vector + 0.1 * torch.randn_like(x_vector)

        negatives_vectors = generate_negative_samples(table, x0_idx, x1_idx)

        for neg_vector in negatives_vectors:
            positive_list.append(pos_vector)
            anchor_list.append(anchor_vector)
            negative_list.append(neg_vector)

    if len(anchor_list) == 0:
        return torch.tensor(0.0, requires_grad=True)

    anchors = torch.stack(anchor_list, dim=0)    
    positives = torch.stack(positive_list, dim=0)
    negatives = torch.stack(negative_list, dim=0)

    loss = loss_fn(anchors, positives, negatives)
    return loss

def compute_batch_boundary_contrastive_loss(tables, batch_pairs, loss_fn):
    batch_size = tables.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        table = tables[i]
        pairs = batch_pairs[i]

        loss_s = compute_boundary_contrastive_loss(table, pairs, loss_fn, mode='S')
        loss_e = compute_boundary_contrastive_loss(table, pairs, loss_fn, mode='E')
        total_loss += (loss_s + loss_e)

    return total_loss / batch_size 
