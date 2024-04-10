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
from feature_graph.model import FeatModel
from torchvision.models import vit_b_16

class GATBase(Module, LogMixin):
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
        super().__init__()

        # model structure
        self.hidden_size = hidden_size
        self.item_embedding_layer, self.item_proj = self._load_embedding(item_vocab, item_embedding, img_hidden_size, hidden_size)
        # User embedding
        self.user_embedding_layer, self.user_proj = self._load_embedding(user_vocab, user_embedding, 10, hidden_size)
        
        # Attribute text embedding
        self.attr_proj = nn.Linear(text_hidden_size, hidden_size)
        # bodyshape embedding
        
        self.bs_embedding_layer, self.bs_proj = self._load_embedding(bodyshape_vocab, bodyshape_embedding, bs_hidden_size, hidden_size)

        self.outfit_proj = nn.Linear(img_hidden_size, hidden_size)
        self.scene_proj = nn.Linear(img_hidden_size, hidden_size)
        # self.outfit_proj = nn.Identity(img_hidden_size, hidden_size)
        # self.scene_proj = nn.Identity(img_hidden_size, hidden_size)

        self.model = FeatModel(hidden_size=hidden_size, num_heads=8, head_size=64, threshold=threshold)
        self.model_os = FeatModel(hidden_size=hidden_size, num_heads=8, head_size=64, threshold=threshold)

        vit  = vit_b_16()
        weight = torch.load("/data/shy/data/vit_weight/vit_b_16-c867db91.pth")
        vit.load_state_dict(weight)
        for p in vit.parameters():
            p.requires_grad=False
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.float16)
        self.vit = vit
        

    
    def _getNodeEmb(self, UBS_graph, IA_graph,
                attr_text: Tensor,
                item_emb_index: Tensor,
                user_emb_index: Tensor,
                bodyshape_emb_index: Tensor,
                outfit_img: Tensor,
                scene_img: Tensor,
                istrain: bool,
                ):
        
        
        item_emb = self.item_proj(self.item_embedding_layer(item_emb_index).to(torch.float32))

        attr_emb = self.attr_proj(attr_text[:, 0, :])
        bs_emb = self.bs_proj(self.bs_embedding_layer(bodyshape_emb_index))
        smpl_emb = self.user_proj(self.user_embedding_layer(user_emb_index).to(torch.float32))
        user_emb = self.user_proj(self.user_embedding_layer(user_emb_index).to(torch.float32))
        device = item_emb.device

        user_node_feat = torch.cat([user_emb, bs_emb, smpl_emb], dim=0)
        user_graph = UBS_graph
        user_selected_index = torch.arange(len(user_emb_index)).to(torch.long).to(device)

        item_node_feat = torch.cat([item_emb, attr_emb], dim=0)
        item_graph = IA_graph
        item_selected_index = torch.arange(len(item_emb_index)).to(torch.long).to(device)

        outfit_emb = self.outfit_proj(self._process_img(outfit_img)).to(torch.float)
        scene_emb = self.scene_proj(self._process_img(scene_img)).to(torch.float)

        if istrain:
            user_global_emb, item_global_emb = self.model.forward_graph(user_node_feat, item_node_feat, (user_graph, user_selected_index), (item_graph, item_selected_index))
            outfit_global_emb, scene_global_emb = self.model_os(outfit_emb, scene_emb)
        else: 
            user_global_emb, item_global_emb = self.model.predict_graph(user_node_feat, item_node_feat, (user_graph, user_selected_index), (item_graph, item_selected_index))
            outfit_global_emb = self.model_os.predict(outfit_emb)
            scene_global_emb = self.model_os.predict(scene_emb)
        

        return user_global_emb, item_global_emb, outfit_global_emb, scene_global_emb

    def _load_embedding(self, vocab, embedding, input_size, output_size):
        vocab = json.load(open(vocab, 'r'))
        # print(embedding)
        embedding = torch.load(embedding, map_location='cpu')
        embedding_layer = nn.Embedding(len(vocab),input_size).from_pretrained(embedding, freeze=False)
        proj = nn.Linear(input_size, output_size)
        return embedding_layer, proj
    
    def _process_img(self, tensor):
        feats = self.vit._process_input(tensor)
        feats = feats.to(torch.float)
        batch_class_token = self.vit.class_token.expand(tensor.shape[0], -1, -1).to(tensor.device)
        feats = torch.cat([batch_class_token, feats], dim=1).to(torch.float16)

        feats = self.vit.encoder(feats)
        return feats