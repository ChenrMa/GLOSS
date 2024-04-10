import json
from PIL import Image
import dgl, os
import torch
from torchvision import transforms
from general_util.logger import get_child_logger
from data_loader.data_utils import EmbeddingMatrix
from torchvision import transforms
import copy
from torchvision.models import ViT_B_16_Weights

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
                 attr_vocab: str,
                 item_vocab: str,
                 node_vocab: str,
                 bodyshape_vocab: str,
                 outfit_vocab: str,
                 scene_vocab: str,
                 imgpathdir: str,
                 scene2path:str,
                 uc_edge: str,
                 cu_edge: str,
                 co_edge: str,
                 oc_edge: str,
                 oi_edge: str,
                 io_edge: str,
                 ia_edge: str,
                 ub_edge: str,
                 cs_edge: str,
                 outfit_location: str,
                 scene_location: str,
                 embedding: EmbeddingMatrix):

        self.user_vocab = json.load(open(user_vocab, 'r'))
        self.attr_vocab = json.load(open(attr_vocab, 'r'))
        self.item_vocab = json.load(open(item_vocab, 'r'))
        self.node_vocab = torch.load(node_vocab)
        self.bodyshape_vocab = json.load(open(bodyshape_vocab, 'r'))
        self.outfit_vocab = json.load(open(outfit_vocab, 'r'))
        self.scene_vocab = json.load(open(scene_vocab, 'r'))
        self.imagepathdir = imgpathdir
        self.scene2imgpath = json.load(open(scene2path))

        self.transform = transforms.Compose(
            [transforms.Resize([224,224]), transforms.ToTensor(),  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        self.ia = json.load(open(ia_edge))
        self.oi = json.load(open(oi_edge))
        self.co = json.load(open(co_edge))
        self.oc = json.load(open(oc_edge))
        self.uc = json.load(open(uc_edge))
        self.ub = json.load(open(ub_edge))
        self.item_id2vocab = {}
        for item, id in self.item_vocab.items():
            self.item_id2vocab[id] = item
        self.attr_id2vocab = {}
        for attr, id in self.attr_vocab.items():
            self.attr_id2vocab[id] = attr
        self.outfit_id2vocab = {}
        self.user_id2vocab = {}
        for user, id in self.user_vocab.items():
            self.user_id2vocab[id] = user
        self.scene_id2vocab = {}
        self.bodyshape_id2vocab = {}
        for bs, id in self.bodyshape_vocab.items():
            self.bodyshape_id2vocab[id] = bs
        self.combination_id2vocab = {}

        self.k = 1

        self.node2type = {}
        for k, v_ls in self.node_vocab.items():
            # print(k, v_ls)
            for v in v_ls:
                if v not in self.node2type:
                    self.node2type[v] = k
                    ...
        assert 'u' in self.node_vocab
        assert 'i' in self.node_vocab
        assert 'a' in self.node_vocab
        assert 'b' in self.node_vocab
        self.embedding = embedding
        
        self.cu = json.load(open(cu_edge))
        self.io = json.load(open(io_edge))
        self.outfit_location = json.load(open(outfit_location))
        self.scene_location = json.load(open(scene_location))
        self.cs = json.load(open(cs_edge))

        self.preprocessing = ViT_B_16_Weights.DEFAULT.transforms()

        
        
    def _get_emb_index(self, base_arr, data_dict, vocab):
        data_arr = []
        emb_index = []
        for b in base_arr:
            if data_dict.__contains__(b):
                data_arr.extend(data_dict[b])
        
        data_arr = list(set(data_arr))

        for d in data_arr:
            emb_index.append(vocab[d])
        return data_arr, emb_index
    
    def _get_graph(self, src_list, dst_list, edge_dict):
        nodes = src_list + dst_list
        node_dict = {}
        for i, n in enumerate(nodes):
            node_dict[n] = i

        row = []
        col = []
        for s in src_list:
            for t in edge_dict[s]:
                if dst_list.__contains__(t):
                    row.append(node_dict[s])
                    col.append(node_dict[t])
        
        graph = dgl.graph((row, col), num_nodes=len(nodes))
        graph = dgl.add_self_loop(graph)
        return graph

    def _get_emb_patch(self, data_list, location_dir, emb_size=768):
        emb = torch.zeros((len(data_list), 197, emb_size))
        for i, d in enumerate(data_list):
            emb[i,] = torch.load(os.path.join(location_dir,f"{d}.pt"))
        return emb


    def __call__(self, batch):
        outfit_list = [*batch]

        new_outfit = []
        for o in outfit_list:
            if self.outfit_location.__contains__(o):
                new_outfit.append(o)
        scene_list = []
        new_outfit_list = []
        for o in new_outfit:
            comb = self.oc[o][0]
            if self.cs.__contains__(comb):
                scene = self.cs[comb][0]
                if self.scene_location.__contains__(scene):
                    scene_list.append(scene)
                    new_outfit_list.append(o)
        # print(outfit_list, new_outfit, new_outfit_list)
        outfit_list = new_outfit_list
        scene_img_list = self.getImageList(scene_list, self.scene2imgpath)
        outfit_img_list = self.getImageListByDir(outfit_list, "outfit")

        item_arr, item_emb_index = self._get_emb_index(outfit_list, self.oi, self.item_vocab)
        attr_arr, attr_emb_index = self._get_emb_index(item_arr, self.ia, self.attr_vocab)

        ia_graph = self._get_graph(item_arr, attr_arr, self.ia)

        comb_arr = []
        for o in outfit_list:
            comb_arr.extend(self.oc[o])
        
        user_arr, user_emb_index = self._get_emb_index(comb_arr, self.cu, self.user_vocab)
        bs_arr, bs_emb_index = self._get_emb_index(user_arr, self.ub, self.bodyshape_vocab)

        ub = copy.deepcopy(self.ub)
        for i, u in enumerate(user_arr):
            bs_arr.append(f"smpl{i}")
            ub[u].append(f"smpl")
        ub_graph = self._get_graph(user_arr, bs_arr, ub)

        # print(outfit_list, item_emb_index, user_emb_index, bs_emb_index)
        

        item_emb_index = torch.tensor(item_emb_index, dtype=torch.long)
        attr_emb_index = torch.tensor(attr_emb_index, dtype=torch.long)
        user_emb_index = torch.tensor(user_emb_index, dtype=torch.long)
        bs_emb_index = torch.tensor(bs_emb_index, dtype=torch.long)

        outfit_emb_index = torch.tensor([self.outfit_vocab[o] for o in new_outfit_list], dtype=torch.long)
        scene_emb_index = torch.tensor([self.scene_vocab[s] for s in scene_list], dtype=torch.long)

        return {
            "UBS_graph": ub_graph,
            "IA_graph": ia_graph,
            "attr_text": torch.index_select(self.embedding.attr_text, dim=0, index=attr_emb_index),
            "item_emb_index": item_emb_index,
            "user_emb_index": user_emb_index,
            "bodyshape_emb_index": bs_emb_index,
            "outfit_img": outfit_img_list,
            "scene_img": scene_img_list,
            "outfit_emb_index": outfit_emb_index,
            "scene_emb_index": scene_emb_index,
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
        img_list = self.preprocessing(img_list)
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
