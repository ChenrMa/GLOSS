import json, sys
import dgl
import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import Module
import os
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

def sign(x):
    """Return hash code of x."""

    return x.detach().sign()

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
                pos_mask: Tensor,
                neg_mask: Tensor,
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
        
        batch_size= comb_index.size()[0]

        
        pos_score, neg_score = self._getlogits_predict(node_emb, comb_index, pos_outfit_index, neg_outfit_index)  # [batch, num_pos], [batch, num_neg]

        
        all_res = {}
        for k in self.metrics.keys():
            all_res[k] = 0
        
        # print(pos_outfit_index.size(), pos_score.size())
        for i in range(batch_size):
            pos = pos_score[i].tolist()[:pos_mask[i].item()]
            neg = neg_score[i].tolist()[:neg_mask[i].item()]
            # print(len(pos), len(neg))
            for k, func in self.metrics.items():
                all_res[k] += func([pos], [neg])

        for k in all_res.keys():
            all_res[k] /= batch_size

        # res = {
        #     "comb": comb_emb_index,
        #     "pos_score": pos_score,
        #     "neg_score": neg_score,
        # }
        res = {}
        res.update(all_res)

        return res
    
    
    
    
