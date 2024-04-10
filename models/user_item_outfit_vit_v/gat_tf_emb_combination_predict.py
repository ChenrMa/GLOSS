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
                 user_vocab: str,
                 user_embedding: str,
                 bodyshape_vocab: str,
                 bodyshape_embedding: str,
                 text_hidden_size: int = 768,
                 img_hidden_size: int = 2048,
                 bs_hidden_size: int = 8,
                 hidden_size: int = 768,
                 threshold: int=0.5):
        super(GATTransformer, self).__init__(item_vocab, item_embedding, 
                                            user_vocab, user_embedding, 
                                            bodyshape_vocab, bodyshape_embedding,
                                            text_hidden_size, img_hidden_size, bs_hidden_size, hidden_size, 
                                            threshold
                                            )


    def forward(self,
                UBS_graph, IA_graph,
                attr_text: Tensor,
                item_emb_index: Tensor,
                user_emb_index: Tensor,
                bodyshape_emb_index: Tensor,
                outfit_img: Tensor,
                scene_img: Tensor,
                outfit_emb_index: Tensor,
                scene_emb_index: Tensor,
                epoch: int
                ):

        user_global_emb, item_global_emb, outfit_global_emb, scene_global_emb = self._getNodeEmb(UBS_graph, IA_graph, 
                                    attr_text, item_emb_index, user_emb_index, bodyshape_emb_index,
                                    outfit_img, scene_img,
                                    False)

        res = {
            "user_index": user_emb_index,
            "user_emb": user_global_emb,
            "item_index": item_emb_index,
            "item_emb": item_global_emb,
            "outfit_index": outfit_emb_index,
            "outfit_emb": outfit_global_emb,
            "scene_index": scene_emb_index,
            "scene_emb": scene_global_emb
        }
    
        return res
