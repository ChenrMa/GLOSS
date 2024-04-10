import torch
import torch.nn.functional as F
from torch import nn
from general_util.InfoNCE_loss import InfoNCE

class SampleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = InfoNCE(negative_mode="paired")
    
    def forward(self, anchor, sample):
        # print(anchor.size())
        num = anchor.size()[0]
        anchor_row = anchor.repeat_interleave(num,0)
        anchor_col = anchor.repeat(num, 1)
        # print(anchor_row.size(), anchor_col.size())
        # print(anchor_row)
        # print(anchor_col)
        anchor_sim = torch.cosine_similarity(anchor_row, anchor_col , dim=1).reshape((num, num))
        
        pos_values, pos_indices = anchor_sim.topk(5, dim=1, largest=True, sorted=True)
        neg_values, neg_indices = anchor_sim.topk(10, dim=1, largest=False, sorted=True)
        # print(pos_indices, neg_indices)

        query_logit = sample.repeat(5, 1)
        pos_indices = pos_indices.reshape(-1)
        pos_logit = torch.index_select(sample, dim=0, index=pos_indices)
        neg_indices = neg_indices.reshape(-1)
        neg_logit = torch.index_select(sample, dim=0, index=neg_indices).reshape((num, 10, -1))
        neg_logit = neg_logit.repeat_interleave(5, 0)
        # print(query_logit.size(), pos_logit.size(), neg_logit.size())

        return self.loss(query_logit, pos_logit, neg_logit)
