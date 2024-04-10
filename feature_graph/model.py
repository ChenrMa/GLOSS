import torch
import torch.nn as nn
from models.graphGenerator import GraphGenerator
from models.gat import GAT
from models.modeling_utils import init_weights
import dgl
from torch.cuda.amp import autocast

class FeatModel(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, head_size=64, threshold=0.1):
        super().__init__()
        self.gat = GAT(num_layers=4, input_size=hidden_size, num_heads=num_heads, head_size=head_size)
        self.gat.apply(init_weights)

        self.graph_structure = GraphGenerator(hidden_size, threshold=threshold)
        print("Feat model: ", threshold)

    def forward(self, l_feat, r_feat):
        batch_size, patch_size = l_feat.size()[:2][:]
        node_l, node_r = l_feat.flatten(start_dim=0, end_dim=1), r_feat.flatten(start_dim=0, end_dim=1)
        
        node_feat = self._graph_structure(node_l, node_r)

        node_feat = self._gather(node_feat, batch_size, patch_size)

        node_feat_l = node_feat[:l_feat.size()[0]]
        node_feat_r = node_feat[l_feat.size()[0]:]

        return node_feat_l, node_feat_r
    
    def _graph_structure(self, node_l, node_r):
        num_l, num_r = len(node_l), len(node_r)
        
        res = self.graph_structure(node_l, node_r)
        fh_l, fh_y = res[:]
        
        device = node_l.device

        lr_graph = dgl.graph((fh_l, fh_y+num_l), num_nodes=num_l + num_r)
        lr_graph = dgl.add_self_loop(lr_graph)
        lr_graph = lr_graph.to(device)

        node_feat = torch.cat([node_l, node_r]).to(dtype=torch.float)
        with autocast(False):
            node_feat = self.gat(lr_graph, node_feat)
        
        return node_feat

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
    
    def predict(self, feat):
        batch_size, patch_size = feat.size()[:2][:]
        node_feat = feat.flatten(start_dim=0, end_dim=1)

        return self._gather(node_feat, batch_size, patch_size)
    
    def forward_graph(self,  l_feat, r_feat, l_graph_class, r_graph_class):
        node_feat = self._graph_structure(l_feat, r_feat)

        node_feat_l = node_feat[:len(l_feat)]
        node_feat_r = node_feat[len(l_feat):]

        l_graph, l_selected = l_graph_class[:]
        r_graph, r_selected = r_graph_class[:]

        global_node_feat_l = self._gather_graph(node_feat_l, l_graph, l_selected)
        global_node_feat_r = self._gather_graph(node_feat_r, r_graph, r_selected)
        return global_node_feat_l, global_node_feat_r
    
    def predict_graph(self,  l_feat, r_feat, l_graph_class, r_graph_class):
        l_graph, l_selected = l_graph_class[:]
        r_graph, r_selected = r_graph_class[:]

        global_node_feat_l = self._gather_graph(l_feat, l_graph, l_selected)
        global_node_feat_r = self._gather_graph(r_feat, r_graph, r_selected)
        return global_node_feat_l, global_node_feat_r


    def _gather_graph(self, feat, graph, selected_index):
        feat = feat.to(torch.float)
        with autocast(False):
            node_feat = self.gat(graph, feat)
        return torch.index_select(node_feat, dim=0, index=selected_index )




        