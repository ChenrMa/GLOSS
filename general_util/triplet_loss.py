import torch
import torch.nn as nn
class TripletLoss(nn.Module):
    def __init__(self, alpha_p, alpha_n):
        super(TripletLoss, self).__init__()
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        return
    
    def forward(self, anchor, positive, negative):
        matched = torch.pow(func.pairwise_distance(anchor, positive), 2)
        mismatched = torch.pow(func.pairwise_distance(anchor, negative), 2)
        part1 = torch.clamp(matched- self.alpha_p, min=0)
        part2 = torch.clamp(self.alpha_n - mismatched, min=0)
        loss = part1 + part2
        loss = torch.mean(loss)
        return loss
    
    def forward_pair(self, pos_logits, neg_logits):
        part1 = torch.clamp(pos_logits- self.alpha_p, min=0)
        part2 = torch.clamp(self.alpha_n - neg_logits, min=0)
        loss = part1 + part2
        loss = torch.mean(loss)
        return loss