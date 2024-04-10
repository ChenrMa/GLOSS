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
                 user_vocab: str,
                 user_embedding: str,
                 bodyshape_vocab: str,
                 bodyshape_embedding: str,
                 text_hidden_size: int = 768,
                 img_hidden_size: int = 2048,
                 bs_hidden_size: int = 8,
                 hidden_size: int = 768,
                 alpha: list=[1,1,1],
                 threshold: int=0.5):
        super(GATTransformer, self).__init__(item_vocab, item_embedding, 
                                            user_vocab, user_embedding, 
                                            bodyshape_vocab, bodyshape_embedding,
                                            text_hidden_size, img_hidden_size, bs_hidden_size, hidden_size, 
                                            threshold,
                                            )

        self.loss_fn = InfoNCE(negative_mode="paired")
        self.loss_os = InfoNCE()
        self.alpha= alpha


    def forward(self,
                UBS_graph, IA_graph,
                attr_text: Tensor,
                item_emb_index: Tensor,
                user_emb_index: Tensor,
                bodyshape_emb_index: Tensor,
                batch_ui: Tensor,
                batch_o_ii: Tensor,
                outfit_img: Tensor,
                scene_img: Tensor,
                batch_oi: Tensor,
                epoch: int
                ):

        user_global_emb, item_global_emb, outfit_global_emb, scene_global_emb = self._getNodeEmb(UBS_graph, IA_graph, 
                                    attr_text, item_emb_index, user_emb_index, bodyshape_emb_index,
                                    outfit_img, scene_img,
                                    True)
        # print(user_global_emb.size(), item_global_emb.size(), outfit_global_emb.size(), scene_global_emb.size())

        # cl user loss
        loss_ui = self._cl_loss(batch_ui, user_global_emb, item_global_emb)
        loss_oi = self._cl_loss(batch_oi, outfit_global_emb, item_global_emb)
        loss_os = self.loss_os(outfit_global_emb, scene_global_emb )
        all_loss =  loss_ui * self.alpha[0]  + loss_oi * self.alpha[1]  + loss_os * self.alpha[2]

        res = {"loss": all_loss, "loss_ui": loss_ui, "loss_oi": loss_oi, "loss_os": loss_os}
    
        return res

    def _cl_loss(self, batch_matrix, l_emb, r_emb):
        pos_index = torch.where(batch_matrix >= 1)
        neg_index_list = [torch.where(batch_matrix[i]<1)[0] for i in range(len(batch_matrix))]

        negative_index = []
        available_index = []
        for i in range(len(pos_index[0])):
            p_index = pos_index[0][i]
            if len(neg_index_list[p_index]) < 10:
                continue
            r_index = torch.randint(len(neg_index_list[p_index]), (10,))
            neg_samples = neg_index_list[p_index][r_index]
            negative_index.append(neg_samples)
            available_index.append(i)
        
        all_loss = 0
        # print(pos_index, neg_index_list)
        if len(available_index) > 0:
            query = torch.index_select(l_emb, dim=0, index=pos_index[0][available_index])
            positive = torch.index_select(r_emb, dim=0, index=pos_index[1][available_index])
            negative_index = torch.cat(negative_index, dim=0)
            negative = torch.index_select(r_emb, dim=0, index=negative_index).reshape((len(pos_index[0][available_index]), 10, -1))
        
            # print(query.size(), positive.size(), negative.size())
            cl_loss = self.loss_fn(query, positive, negative)

            all_loss += cl_loss 
        else:
            return 0
        return cl_loss

    
    
    

