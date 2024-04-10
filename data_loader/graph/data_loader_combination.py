import glob
import json
import os
from collections import defaultdict
from typing import Dict, Any, Union, Callable
import dgl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix
import numpy as np
logger = get_child_logger('Dataset')

class SubgraphDataset(Dataset):
    def __init__(self, triple_file: str, split: str, embedding: EmbeddingMatrix, train_subgraph_dir, test_subgraph_dir, val_subgraph_dir):
        logger.info(f'Loading data file from {triple_file}.')
        self.split = split
        self.triples = json.load(open(triple_file, 'r'))
        if split == 'train':
            self.dir = train_subgraph_dir
        elif split == 'test':
            self.dir = test_subgraph_dir
        elif split == 'val':
            self.dir = val_subgraph_dir

    def __getitem__(self, index) -> T_co:

        # triple = [user, pos_outfit, neg_outfit]
        triple = self.triples[index]
        all_nodes = set()
        all_dgl_graph, all_src, all_dst, all_node2re_id, all_re_id2node = [], [], [], [], []

        if self.split == 'train':
            tri_graph = torch.load(os.path.join(self.dir, '{}_{}_{}.pt'.format(triple[0], triple[1], triple[2])))
        else:
            tri_graph = torch.load(os.path.join(self.dir, '{}_{}_{}.pt'.format(triple["combination"], triple["neg"][0], len(triple["neg"]))))
        
        CO_OI_IA_UC_BU_SC = self._load_subgraph(tri_graph)
        for each in CO_OI_IA_UC_BU_SC:
            all_dgl_graph.append(each[0])
            all_src.append(each[1])
            all_dst.append(each[2])
            all_node2re_id.append(each[3])
            all_re_id2node.append(each[4])
            all_nodes.update(each[5])

        return all_dgl_graph, all_node2re_id, all_re_id2node, list(all_nodes), triple

    def __len__(self):
        return len(self.triples)

    def _load_subgraph(self, tri_graph: dict):

        CO_nodes, CO_orig_edges = tri_graph['CO_nodes'], tri_graph['CO_graph']
        OI_nodes, OI_orig_edges = tri_graph['OI_nodes'], tri_graph['OI_graph']
        UC_nodes, UC_orig_edges = tri_graph['UC_nodes'], tri_graph['UC_graph']
        SC_nodes, SC_orig_edges = tri_graph['SC_nodes'], tri_graph['SC_graph']

        co_node2re_id, co_re_id2node, co_mapped_src, co_mapped_dst = self._graph_process(CO_nodes, CO_orig_edges)
        oi_node2re_id, oi_re_id2node, oi_mapped_src, oi_mapped_dst = self._graph_process(OI_nodes, OI_orig_edges)
        uc_node2re_id, uc_re_id2node, uc_mapped_src, uc_mapped_dst = self._graph_process(UC_nodes, UC_orig_edges)
        sc_node2re_id, sc_re_id2node, sc_mapped_src, sc_mapped_dst = self._graph_process(SC_nodes, SC_orig_edges)
        

        if self.split != "train":
            co_dgl_graph = dgl.graph((torch.from_numpy(np.array(co_mapped_src + co_mapped_dst).astype(np.int64)), torch.from_numpy(np.array(co_mapped_src + co_mapped_dst).astype(np.int64))))
        else:
            co_dgl_graph = dgl.graph((torch.tensor(co_mapped_src + co_mapped_dst), torch.tensor(co_mapped_dst + co_mapped_src)))
        oi_dgl_graph = dgl.graph((torch.tensor(oi_mapped_src + oi_mapped_dst), torch.tensor(oi_mapped_dst + oi_mapped_src)))
        uc_dgl_graph = dgl.graph((torch.tensor(uc_mapped_src + uc_mapped_dst), torch.tensor(uc_mapped_dst + uc_mapped_src)))
        sc_dgl_graph = dgl.graph((torch.tensor(sc_mapped_src + sc_mapped_dst), torch.tensor(sc_mapped_dst + sc_mapped_src)))
        
        CO = [co_dgl_graph, co_mapped_src, co_mapped_dst, co_node2re_id, co_re_id2node, CO_nodes]
        OI = [oi_dgl_graph, oi_mapped_src, oi_mapped_dst, oi_node2re_id, oi_re_id2node, OI_nodes]
        UC = [uc_dgl_graph, uc_mapped_src, uc_mapped_dst, uc_node2re_id, uc_re_id2node, UC_nodes]
        SC = [sc_dgl_graph, sc_mapped_src, sc_mapped_dst, sc_node2re_id, sc_re_id2node, SC_nodes]
        return CO, OI, UC,  SC,

    def _graph_process(self, nodes, orig_edges):
        node2re_id = {}
        re_id2node = {}
        for i, node in enumerate(nodes):
            node2re_id[node] = i
            re_id2node[i] = node

        mapped_src = []
        mapped_dst = []
        for e in orig_edges:
            mapped_src.append(node2re_id[e[0]])
            mapped_dst.append(node2re_id[e[1]])
        return node2re_id, re_id2node, mapped_src, mapped_dst
    
    def _graph_comb(self, nodes_list, orig_edges_list):        
        edge_dict_list = []
        for orig_edges in orig_edges_list:
            edge_dict = {}
            for e in orig_edges:
                if not edge_dict.__contains__(e[0]):
                    edge_dict[e[0]] = []
                edge_dict[e[0]].append(e[1])

                if not edge_dict.__contains__(e[1]):
                    edge_dict[e[1]] = []
                edge_dict[e[1]].append(e[0])
            edge_dict_list.append(edge_dict)
        
        node2re_id = {}
        re_id2node = {}
        mapped_src = []
        mapped_dst = []

        def add_edge(e0, e1):
            if not node2re_id.__contains__(e0):
                idx = len(node2re_id)
                node2re_id[e0] = idx
                re_id2node[idx] = e0
            if not node2re_id.__contains__(e1):
                idx = len(node2re_id)
                node2re_id[e1] = idx
                re_id2node[idx] = e1
            
            mapped_src.append(node2re_id[e0])
            mapped_dst.append(node2re_id[e1])

        num_view = len(edge_dict_list)
        for i in range(num_view):
            for j in range(i, num_view):
                for edge in orig_edges_list[i]:
                    if edge_dict_list[j].__contains__(edge[0]):
                        for e in edge_dict_list[j][edge[0]]:
                            add_edge(edge[1], e)
                    if edge_dict_list[j].__contains__(edge[1]):
                        for e in edge_dict_list[j][edge[1]]:
                            add_edge(edge[0], e)
        
        return node2re_id, re_id2node, mapped_src, mapped_dst

        
        



