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
    def __init__(self, triple_file, split, embedding):
        logger.info(f'Loading data file from {triple_file}.')
        triples = json.load(open(triple_file, 'r'))
        self.triples = []
        for c, o_arr in triples.items():
            self.triples.extend(o_arr)
        self.triples = list(set(self.triples))

    def __getitem__(self, index) -> T_co:
        outfit = self.triples[index]
        return outfit

    def __len__(self):
        return len(self.triples)

    def _load_subgraph(self, tri_graph: dict):

        IA_nodes, IA_orig_edges = tri_graph['IA_nodes'], tri_graph['IA_graph']
        BU_nodes, BU_orig_edges = tri_graph['BU_nodes'], tri_graph['BU_graph']
        SU_nodes, SU_orig_edges = tri_graph['SU_nodes'], tri_graph['SU_graph']

        ia_node2re_id, ia_re_id2node, ia_mapped_src, ia_mapped_dst = self._graph_process(IA_nodes, IA_orig_edges)
        ubs_node2re_id, ubs_re_id2node, ubs_mapped_src, ubs_mapped_dst = self._graph_comb([BU_nodes, SU_nodes], [BU_orig_edges, SU_orig_edges])
        
        ia_dgl_graph = dgl.graph((torch.tensor(ia_mapped_src + ia_mapped_dst), torch.tensor(ia_mapped_dst + ia_mapped_src)))
        ubs_dgl_graph = dgl.graph((torch.tensor(ubs_mapped_src + ubs_mapped_dst), torch.tensor(ubs_mapped_dst + ubs_mapped_src)))
        
        IA = [ia_dgl_graph, ia_mapped_src, ia_mapped_dst, ia_node2re_id, ia_re_id2node, IA_nodes]
        UBS = [ubs_dgl_graph, ubs_mapped_src, ubs_mapped_dst, ubs_node2re_id, ubs_re_id2node, ubs_node2re_id.keys()]
        return UBS, IA 

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

        
        



