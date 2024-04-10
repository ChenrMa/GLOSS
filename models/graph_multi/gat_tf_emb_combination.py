import json, sys
import dgl
import numpy as np
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module
from general_util.logger import get_child_logger
from general_util.mixin import LogMixin
from models.modeling_utils import init_weights, initialize_vision_backbone
from general_util.metrics import ROC, BA, RECALL
logger = get_child_logger('GATTransformerVocab')
from general_util.cl_loss import AllLoss
from collections import OrderedDict
from models.graphGenerator import GraphGenerator
from models.gat import GAT
from .gat_base import GATBase
from general_util.InfoNCE_loss import InfoNCE
from general_util.triplet_loss import TripletLoss

def contrastive_loss(margin, im, s):
    size, dim = im.shape
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # for data parallel, reshape to (size, 1)
    return vse_loss / (size - 1)

def sign(x):
    """Return hash code of x."""

    return x.detach().sign()

def cal_similarity_loss(batch_s, emb):

    fi = torch.cosine_similarity(emb.unsqueeze(0), emb.unsqueeze(1), dim=-1) # len IXI
    similarity_loss = -torch.sum(batch_s * fi - torch.log(torch.ones_like(fi) + torch.exp(fi)))
    similarity_loss /= len(emb)
    return similarity_loss

class GATTransformer(GATBase):
    def __init__(self,
                 item_vocab: str,
                 item_embedding: str,
                 item_graph_embedding: str,
                 outfit_vocab: str,
                 outfit_embedding: str,
                 outfit_graph_embedding: str,
                 user_vocab: str,
                 user_embedding: str,
                 user_graph_embedding: str,
                 combination_vocab: str,
                 combination_embedding: str,
                 scene_vocab: str,
                 scene_embedding: str,
                 scene_graph_embedding: str,
                 img_hidden_size: int = 2048,
                 hidden_size: int = 768,
                 hash_hidden_size : int = 256,
                 alpha: int = 0.5,
                 iter_ids: list=[1, 4, 3, 4, 1],
                 graph_alpha: int=0.1,
                 gnn: Module = None):
        super(GATTransformer, self).__init__(item_vocab, item_embedding, item_graph_embedding,
                                            outfit_vocab, outfit_embedding, outfit_graph_embedding,
                                            user_vocab, user_embedding, user_graph_embedding, 
                                            combination_vocab, combination_embedding,
                                            scene_vocab, scene_embedding, scene_graph_embedding,
                                            img_hidden_size,hidden_size, hash_hidden_size,
                                             iter_ids, graph_alpha,
                                            gnn)

        self.cl_user = InfoNCE()

        self.triplet_loss = TripletLoss(1, 1)
        self.alpha = alpha


    def forward(self,
                CO_graph,
                OI_graph,
                UC_graph,
                SC_graph,
                CO_input_emb_index,
                OI_input_emb_index,
                UC_input_emb_index,
                SC_input_emb_index,
                comb_index: Tensor,
                pos_outfit_index: Tensor,
                neg_outfit_index: Tensor,
                user_index: Tensor,
                scene_index: Tensor,
                item_emb_index: Tensor,
                outfit_emb_index: Tensor,
                user_emb_index: Tensor,
                scene_emb_index: Tensor,
                comb_emb_index: Tensor,
                epoch: int
                ):

        node_emb = self._getNodeEmb(CO_graph, OI_graph, UC_graph, SC_graph,
                                    CO_input_emb_index,OI_input_emb_index,UC_input_emb_index,SC_input_emb_index,
                                   item_emb_index, outfit_emb_index, user_emb_index, scene_emb_index,comb_emb_index)

        pos_logits, neg_logits = self._getlogits(node_emb, comb_index, pos_outfit_index, neg_outfit_index)  # [batch, 1]

        pos_outfit = torch.index_select(node_emb, 0, index=pos_outfit_index)
        
        attn_loss = self.cl_user(torch.index_select(node_emb, 0, index=comb_index),pos_outfit) \
                    + self.cl_user(torch.index_select(node_emb, 0, index=user_index), pos_outfit) \
                    +self.cl_user(torch.index_select(node_emb, 0, index=scene_index), pos_outfit)

        # bpr loss
        x = pos_logits - neg_logits
        bprloss = -torch.mean(torch.log(torch.sigmoid(x)))
   
        all_loss = bprloss  + attn_loss * self.alpha
        ret = {"loss": all_loss,"attn_loss": attn_loss}        

        all_res = {}
        for k, func in self.metrics.items():
            all_res[k] = func(pos_logits.tolist(), neg_logits.tolist())
        
        ret.update(all_res)
        # print(ret)
        return ret

    
    

