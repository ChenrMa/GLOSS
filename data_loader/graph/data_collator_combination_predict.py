import json
from PIL import Image
import dgl, os
import torch
from torchvision import transforms
from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix

logger = get_child_logger('SubgraphCollatorVocab')

def get_s(src_emb_index, dst_emb_index, src_id2vocab, dst_id2vocab, edge, k):
    SD = torch.zeros((len(src_emb_index), len(dst_emb_index)))
    for i, src in enumerate(src_emb_index.tolist()):
        for j, dst in enumerate(dst_emb_index.tolist()):
            if src_id2vocab[src] in edge.keys():
                if dst_id2vocab[dst] in edge[src_id2vocab[src]]:
                    SD[i][j] = 1
    SS = torch.matmul(SD, SD.t())
    SS_one = torch.ones_like(SS)
    SS_zero = torch.zeros_like(SS)
    batch_ss = torch.where(SS >= k, SS_one, SS_zero)
    return batch_ss

class SubgraphCollatorVocab:
    def __init__(self,
                 user_vocab: str,
                 outfit_vocab: str,
                 item_vocab: str,
                 combination_vocab: str,
                 node_vocab: str,
                 scene_vocab: str,
                 imgpathdir: str,
                 scene2path: str,
                 uc_edge: str,
                 cu_edge: str,
                 co_edge: str,
                 oi_edge: str,
                 sc_edge: str,
                 cs_edge: str,
                 embedding: EmbeddingMatrix):

        self.user_vocab = json.load(open(user_vocab, 'r'))
        self.outfit_vocab = json.load(open(outfit_vocab, 'r'))
        self.item_vocab = json.load(open(item_vocab, 'r'))
        self.combination_vocab = json.load(open(combination_vocab, 'r'))
        self.node_vocab = torch.load(node_vocab)
        self.scene_vocab = json.load(open(scene_vocab, 'r'))
        self.imagepathdir = imgpathdir
        self.scene2imgpath = json.load(open(scene2path))
        self.scene2imgpath_kset = set(self.scene2imgpath.keys())

        self.transform = transforms.Compose(
            [transforms.Resize([224,224]), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.oi = json.load(open(oi_edge))
        self.co = json.load(open(co_edge))
        self.uc = json.load(open(uc_edge))
        self.sc = json.load(open(sc_edge))
        self.item_id2vocab = {}
        for item, id in self.item_vocab.items():
            self.item_id2vocab[id] = item
        self.outfit_id2vocab = {}
        for outfit, id in self.outfit_vocab.items():
            self.outfit_id2vocab[id] = outfit
        self.user_id2vocab = {}
        for user, id in self.user_vocab.items():
            self.user_id2vocab[id] = user
        self.scene_id2vocab = {}
        for scene, id in self.scene_vocab.items():
            self.scene_id2vocab[id] = scene
        self.combination_id2vocab = {}
        for comb, id in self.combination_vocab.items():
            self.combination_id2vocab[id] = comb

        self.k = 1

        self.node2type = {}
        for k, v_ls in self.node_vocab.items():
            # print(k, v_ls)
            for v in v_ls:
                if v not in self.node2type:
                    self.node2type[v] = k
                else:
                    print("conflict: ", self.node2type[v], k)
                    # assert self.node2type[v] == k, (self.node2type[v], k)  # Check repetition.
                    ...
        assert 'u' in self.node_vocab
        assert 'o' in self.node_vocab
        assert 'i' in self.node_vocab
        assert 'b' in self.node_vocab
        assert 's' in self.node_vocab
        self.embedding = embedding
        
        self.cu = json.load(open(cu_edge))
        self.cs = json.load(open(cs_edge))


    def __call__(self, batch):
        all_dgl_graph, all_node2re_id, all_re_id2node, all_nodes, all_triples = zip(*batch)
        # print(all_nodes)
        batch_size = len(all_dgl_graph)
        _nodes = set()
        _node2emb = {}

        max_subgraph_num = len(all_dgl_graph[0])

        for b in range(batch_size):
            _nodes.update(all_nodes[b])
        
        _nodes = list(_nodes)
        _nodes.sort()

        users = []
        user_emb_index = []
        outfits = []
        outfit_emb_index = []
        items = []
        item_emb_index = []
        combinations = []
        comb_emb_index = []
        scenes = []
        scene_emb_index = []
        # print("node2type", len(self.node2type))
        for _node in _nodes:
            _node_type = self.node2type[_node]
            # print(_node, _node_type)
            if _node_type == 'i':
                items.append(_node)
                item_emb_index.append(self.item_vocab[_node])
            elif _node_type == 'u':
                users.append(_node)
                user_emb_index.append(self.user_vocab[_node])
            elif _node_type == 'o':
                outfits.append(_node)
                outfit_emb_index.append(self.outfit_vocab[_node])
            elif _node_type == 'c':
                combinations.append(_node)
                comb_emb_index.append(self.combination_vocab[_node])
            elif _node_type == 's':
                scenes.append(_node)
                scene_emb_index.append(self.scene_vocab[_node])
            else:
                raise RuntimeError(f"Unrecognized node type: {_node_type}.")

        item_emb_index = torch.tensor(item_emb_index, dtype=torch.long)
        outfit_emb_index = torch.tensor(outfit_emb_index, dtype=torch.long)
        user_emb_index = torch.tensor(user_emb_index, dtype=torch.long)
        scene_emb_index = torch.tensor(scene_emb_index, dtype=torch.long)
        comb_emb_index = torch.tensor(comb_emb_index, dtype=torch.long)
        
        node2emb_index = {}
        for i, _node in enumerate(users + outfits + items + scenes + combinations):
            node2emb_index[_node] = i

        

        comb_index, pos_outfit_index, neg_outfit_index = [], [], []
        user_index, scene_index = [], []
        pad = 0
        pos_len = 10
        neg_len = 10
        all_pos_mask = []
        all_neg_mask = []
        for b in range(batch_size):
            triple = all_triples[b]
            comb, o_pos_list, o_neg_list = triple["combination"], triple["pos"], triple["neg"]
            comb_index.append(node2emb_index[comb])
            user_index.append(node2emb_index[self.cu[comb][0]])
            scene_index.append(node2emb_index[self.cs[comb][0]])
            tem_pos = [node2emb_index[o] for o in o_pos_list]
            all_pos_mask.append(len(tem_pos))
            # if len(tem_pos) < pos_len:
            #     tem_pos.extend([pad]*(pos_len - len(tem_pos)))
            pos_outfit_index.append(torch.tensor(tem_pos))
            # print(triple)
            try:
                tem_neg = [node2emb_index[o] for o in o_neg_list]
            except Exception as e:
                print(comb, o_pos_list, o_neg_list)
                print(outfits)
                raise e
            all_neg_mask.append(len(tem_neg))
            neg_outfit_index.append(torch.tensor(tem_neg))

            # print(o_pos_list, o_neg_list, len(tem_pos), len(tem_neg))

        comb_index = torch.tensor(comb_index)
        pos_outfit_index = torch.stack(pos_outfit_index, dim=0)
        neg_outfit_index = torch.stack(neg_outfit_index, dim=0)
        all_pos_mask = torch.tensor(all_pos_mask)
        all_neg_mask = torch.tensor(all_neg_mask)
        user_index = torch.tensor(user_index)
        scene_index = torch.tensor(scene_index)

        # print(pos_outfit_index.size(), neg_outfit_index.size())

        tri_graph, tri_input_emb_index = [], []
        for l in range(max_subgraph_num):
            all_graphs, all_input_emb_index = [], []
            for b in range(batch_size):
                all_graphs.append(all_dgl_graph[b][l])
                sg_node_num = len(all_re_id2node[b][l])
                sg_input_emb_index = list(map(lambda x: node2emb_index[all_re_id2node[b][l][x]], range(sg_node_num)))
                all_input_emb_index.extend(sg_input_emb_index)
            graph = dgl.batch(all_graphs)
            input_emb_index = torch.tensor(all_input_emb_index, dtype=torch.long)
            tri_graph.append(graph)
            tri_input_emb_index.append(input_emb_index)

        user_input_emb_index = list(map(lambda x: node2emb_index[users[x]], range(len(users))))
        user_input_emb_index = torch.tensor(user_input_emb_index, dtype=torch.long)

        return {
            "CO_graph": tri_graph[0],
            "OI_graph": tri_graph[1],
            "UC_graph": tri_graph[2],
            "SC_graph": tri_graph[3],
            "CO_input_emb_index": tri_input_emb_index[0],
            "OI_input_emb_index": tri_input_emb_index[1],
            "UC_input_emb_index": tri_input_emb_index[2],
            "SC_input_emb_index": tri_input_emb_index[3],
            "comb_index": comb_index,
            "user_index": user_index,
            "scene_index": scene_index,
            "pos_outfit_index": pos_outfit_index,
            "neg_outfit_index": neg_outfit_index,
            "pos_mask": all_pos_mask,
            "neg_mask": all_neg_mask,
            "item_emb_index": item_emb_index,
            "outfit_emb_index": outfit_emb_index,
            "user_emb_index": user_emb_index,
            "scene_emb_index": scene_emb_index,
            "comb_emb_index": comb_emb_index,
        }

    def getImageList(self, arr, path_set):
        '''
        获取需要的imagelist
        '''
        img_list = []
        for item in arr:
            if item in path_set:
                path = path_set[item]
                with open(path, "rb") as f:
                    img = Image.open(f).convert("RGB")
                img_list.append(self.transform(img))
            else:
                img_list.append(torch.zeros(3, 150, 150))
        img_list = torch.stack(img_list, dim=0)
        return img_list

    def getImageListByDir(self, arr, subdir):
        img_list = []
        for item in arr:
            path = os.path.join(self.imagepathdir, subdir, "{}.png".format(item[1:]))
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
                img_list.append(self.transform(img))
        img_list = torch.stack(img_list, dim=0)
        return img_list
