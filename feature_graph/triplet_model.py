import torch
import torch.nn as nn
from models.graphGenerator import GraphGenerator
from models.gat import GAT
from models.modeling_utils import init_weights
import dgl
from torch.cuda.amp import autocast

class FeatModel(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.gat = GAT(num_layers=4, input_size=hidden_size, num_heads=12, head_size=64)
        self.gat.apply(init_weights)

        self.graph_structure = GraphGenerator(hidden_size)

    def forward(self, l_feat, r_feat):
        batch_size, patch_size = l_feat.size()[:2][:]
        node_l, node_r = l_feat.flatten(start_dim=0, end_dim=1), r_feat.flatten(start_dim=0, end_dim=1)
        
        num_l, num_r = len(node_l), len(node_r)
        
        try:
            res = self.graph_structure(node_l, node_r)
            fh_l, fh_y = res[:]
        except Exception as e:
            print(node_l, node_r)
            print(node_l.size(), node_r.size())
            print(res)
            raise e
        

        device = l_feat.device

        lr_graph = dgl.graph((fh_l, fh_y+num_l), num_nodes=num_l + num_r)
        lr_graph = dgl.add_self_loop(lr_graph)
        lr_graph = lr_graph.to(device)

        node_feat = torch.cat([node_l, node_r]).to(dtype=torch.float)
        with autocast(False):
            node_feat = self.gat(lr_graph, node_feat)
        return self._gather(node_feat, batch_size, patch_size)
        
    def _gather(self, feat, batch_size, patch_size):
        img_node = torch.arange(0, len(feat), patch_size).to(torch.long)
        row = torch.arange(0, len(feat)).to(torch.long)
        col = img_node.repeat_interleave(patch_size)

        device = feat.device
        graph = dgl.graph((row, col), num_nodes=len(feat))
        graph = dgl.add_self_loop(graph).to(device)

        with autocast(False):
            node_feat = self.gat(graph, feat)
        return torch.index_select(node_feat, dim=0, index=img_node.to(device) )
        