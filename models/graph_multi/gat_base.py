import json, sys
import dgl
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

class GATBase(Module, LogMixin):
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
                 graph_alpha: list=[1,1,1, 1],
                 gnn: Module = None):
        super().__init__()

        # model structure
        self.hidden_size = hidden_size
        self.item_embedding_layer, self.item_proj = self._load_embedding(item_vocab, item_embedding, img_hidden_size, hidden_size)
        self.item_embedding_graph_layer, _ = self._load_embedding(item_vocab, item_graph_embedding, hidden_size, hidden_size)
        # User embedding
        self.user_embedding_layer, self.user_proj = self._load_embedding(user_vocab, user_embedding, 10, hidden_size)
        self.user_graph_embedding_layer, self.user_personalize_proj = self._load_embedding(user_vocab, user_graph_embedding, hidden_size, hidden_size)
        # Outfit embedding
        self.outfit_embedding_layer, self.outfit_proj = self._load_embedding(outfit_vocab, outfit_embedding, img_hidden_size, hidden_size)
        self.outfit_graph_embedding_layer, _ = self._load_embedding(outfit_vocab, outfit_graph_embedding, img_hidden_size, hidden_size)
        
        # scene image encoding
        self.scene_embedding_layer, self.scene_proj = self._load_embedding(scene_vocab, scene_embedding, img_hidden_size, hidden_size)
        self.scene_graph_embedding_layer, _ = self._load_embedding(scene_vocab, scene_graph_embedding, img_hidden_size, hidden_size)

        # comb
        self.comb_embedding_layer, self.comb_proj = self._load_embedding(combination_vocab, combination_embedding, hidden_size, hidden_size)

        self.gat = gnn
        self.gat.apply(init_weights)

        self.hash_hidden = hash_hidden_size
        self.hash_mlp = nn.Linear(hidden_size, self.hash_hidden)

        self.item_fusion = nn.Linear(3*hidden_size, self.hidden_size)

        self.metrics = {"auc": ROC, "ba": BA, "recall@10": RECALL}

        self.iter_ids = iter_ids

        self.user_graph_alpha, self.item_graph_alpha, self.outfit_graph_alpha, self.scene_graph_alpha  = graph_alpha[:]
    
    def _fusion_graph_emb(self, orig_emb, graph_emb, alpha):
        emb = orig_emb * (1-alpha) + graph_emb * alpha
        return emb

    def _getNodeEmb(self, CO_graph,
                OI_graph,
                UC_graph,
                SC_graph,
                CO_input_emb_index,
                OI_input_emb_index,
                UC_input_emb_index,
                SC_input_emb_index,
                item_emb_index: Tensor,
                outfit_emb_index: Tensor,
                user_emb_index: Tensor,
                scene_emb_index: Tensor,
                comb_emb_index: Tensor,
                ):
        
        
        item_emb = self._fusion_graph_emb(orig_emb=self.item_proj(self.item_embedding_layer(item_emb_index)),
                                          graph_emb=self.item_embedding_graph_layer(item_emb_index),
                                          alpha=self.item_graph_alpha)
        
        outfit_emb = self._fusion_graph_emb(orig_emb=self.outfit_proj(self.outfit_embedding_layer(outfit_emb_index)),
                                          graph_emb=self.outfit_graph_embedding_layer(outfit_emb_index),
                                          alpha=self.outfit_graph_alpha)

        comb_emb = self.comb_proj(self.comb_embedding_layer(comb_emb_index))

        scene_emb = self._fusion_graph_emb(orig_emb=self.scene_proj(self.scene_embedding_layer(scene_emb_index)),
                                          graph_emb=self.scene_graph_embedding_layer(scene_emb_index),
                                          alpha=self.scene_graph_alpha)

        user_emb = self._fusion_graph_emb(orig_emb=self.user_proj(self.user_embedding_layer(user_emb_index).to(torch.float)),
                                          graph_emb=self.user_graph_embedding_layer(user_emb_index),
                                          alpha=self.user_graph_alpha)

        # users + outfits + items + scenes + combinations
        node_emb = torch.cat([user_emb, outfit_emb, item_emb, scene_emb, comb_emb])

        node_feat = None
        tri_graph = [CO_graph, OI_graph, UC_graph, SC_graph, ]
        tri_input_emb_index = [CO_input_emb_index, OI_input_emb_index, UC_input_emb_index, SC_input_emb_index]
        ex_tri_input_emb_index = [
            t.unsqueeze(-1).expand(-1, node_emb.size(-1)) for t in tri_input_emb_index
        ]
        iter_ids = self.iter_ids
        node_emb = node_emb.to(dtype=torch.float)
        for iter_id in iter_ids:
            node_feat = torch.index_select(node_emb, dim=0, index=tri_input_emb_index[iter_id])
            node_feat = node_feat.to(dtype=torch.float)
            with autocast(enabled=False):
                node_feat = self.gat(tri_graph[iter_id], node_feat)
            node_emb = torch.scatter(node_emb, dim=0, index=ex_tri_input_emb_index[iter_id], src=node_feat)

        # Map 512 to lower dimensional space
        node_emb = self.hash_mlp(node_emb)
        node_emb = torch.tanh(node_emb)

        return node_emb

    def _getlogits(self, node_emb, anchor_index, pos_index, neg_index):
        anchor_h = torch.index_select(node_emb, 0, index=anchor_index) # [batch, h]
        batch_size = anchor_h.size()[0]
        p_h = torch.index_select(node_emb, 0, index=pos_index)  # [batch, h]
        n_h = torch.index_select(node_emb, 0, index=neg_index)  # [batch, h]
        # Calculate the score of user and positive and negative cases
        pos_logits = torch.sum(anchor_h * p_h, dim=-1, keepdim=True).reshape(batch_size, 1)
        neg_logits = torch.sum(anchor_h * n_h, dim=-1, keepdim=True).reshape(batch_size, 1)

        if torch.isnan(pos_logits).any() or torch.isnan(neg_logits).any():
            print("pos/neg logits nan", anchor_h, p_h, n_h)
            raise RuntimeError("pos/neg logits nan")
        return pos_logits, neg_logits
    
    def _getlogits_predict(self, node_emb, anchor_index, pos_index, neg_index):
        anchor_h = torch.index_select(node_emb, 0, index=anchor_index) # [batch, h]
        batch_size = anchor_h.size()[0]
        p_h = torch.index_select(node_emb, 0, index=pos_index.reshape(batch_size * pos_index.size(-1)))  # [batch, h]
        n_h = torch.index_select(node_emb, 0, index=neg_index.reshape(batch_size * neg_index.size(-1)))  # [batch, h]
        
        anchor_pos = anchor_h.unsqueeze(1).expand(-1, pos_index.size(-1), -1).reshape(batch_size*pos_index.size(-1), -1)
        pos_score = torch.mean(anchor_pos * p_h, dim=-1).reshape(batch_size, pos_index.size(-1))
        anchor_neg = anchor_h.unsqueeze(1).expand(-1, neg_index.size(-1), -1).reshape(batch_size*neg_index.size(-1), -1)
        neg_score = torch.mean(anchor_neg * n_h, dim=-1).reshape(batch_size, neg_index.size(-1))

        
        return pos_score, neg_score
    
    def _load_embedding(self, vocab, embedding, input_size, output_size):
        vocab = json.load(open(vocab, 'r'))
        # print(embedding)
        embedding = torch.load(embedding, map_location='cpu')
        embedding_layer = nn.Embedding(len(vocab),input_size).from_pretrained(embedding, freeze=False)
        proj = nn.Linear(input_size, output_size)
        return embedding_layer, proj

    def _get_new_feature(self, node_emb, node_index, graph, mlp ,selected_index, scatter_index=None):
        node_feat = torch.index_select(node_emb, dim=0, index=node_index)
        node_feat = node_feat.to(dtype=torch.float)
        with autocast(enabled=False):
            node_feat = self.gat(graph, node_feat)
        
        # print("this")
        node_feat = mlp(node_feat).to(dtype=node_emb.dtype)
        # print(node_feat.dtype, node_emb.dtype)
        new_node_emb = torch.scatter(node_emb, dim=0, index=node_index.unsqueeze(-1).expand(-1, node_emb.size(-1)), src=node_feat )
        item_emb = torch.index_select(new_node_emb, dim=0, index=selected_index)
        
        # if scatter_index is not None:
        #     node_emb = torch.scatter(node_emb, dim=0, index=scatter_index, src=node_feat)
        return item_emb
    
    def _multi_view(self, node_emb_list, input_emb_list, graph_list, mlp_list, input_emb_index):
        specific_feature_list = []
        shared_feature_list = []
        num_views = len(input_emb_list)
        specific_mlp_list = mlp_list[: 2*num_views]
        shared_mlp = mlp_list[2*num_views]
        fusion_mlp = mlp_list[-1]

        node_emb, structure_node_emb = node_emb_list[:]
        
        assert len(specific_mlp_list) == num_views *2
        assert len(graph_list) == num_views
        for i in range(num_views):
            spe_feature = self._get_new_feature(node_emb, input_emb_list[i], graph_list[i], specific_mlp_list[i], input_emb_index)
            specific_feature_list.append(spe_feature)

            shared_feature = self._get_new_feature(node_emb, input_emb_list[i], graph_list[i], shared_mlp, input_emb_index)
            shared_feature_list.append(shared_feature)
        
        for i in range(num_views):
            spe_feature = self._get_new_feature(structure_node_emb, input_emb_list[i], graph_list[i], specific_mlp_list[num_views + i], input_emb_index)
            specific_feature_list.append(spe_feature)

            shared_feature = self._get_new_feature(structure_node_emb, input_emb_list[i], graph_list[i], shared_mlp, input_emb_index)
            shared_feature_list.append(shared_feature)
        shared_feature = sum(shared_feature_list)
        
        fusion_feature = torch.cat(specific_feature_list + [shared_feature], dim=-1)

        
        fusion_feature = fusion_mlp(fusion_feature).to(node_emb.dtype)
        with autocast(enabled=False):
            node_emb = torch.scatter(node_emb, dim=0, index=input_emb_index.unsqueeze(-1).expand(-1, node_emb.size(-1)), src=fusion_feature)
        return node_emb, specific_feature_list, shared_feature_list

    def _structure(self, node_emb, graph_generator, l_index, r_index):
        node_l = torch.index_select(node_emb, dim=0, index=l_index)
        node_r = torch.index_select(node_emb, dim=0, index=r_index)
        fh_x, fh_y = graph_generator(node_l, node_r)

        device = node_emb.device

        lr_graph = dgl.graph((fh_x, fh_y+len(node_l)), num_nodes=len(node_l) + len(node_r))
        lr_graph = dgl.add_self_loop(lr_graph)
        lr_graph = lr_graph.to(device)
        node_feat = torch.cat([node_l, node_r]).to(dtype=torch.float)
        with autocast(False):
            node_feat = self.gat2(lr_graph, node_feat)
        input_index = torch.cat([l_index, r_index])
        # print(input_index.size(), input_index)
        output_index = input_index.unsqueeze(-1).expand(-1, node_emb.size(-1))
        node_feat = node_feat.to(dtype=node_emb.dtype)
        return output_index, node_feat