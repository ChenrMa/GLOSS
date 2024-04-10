import glob
import json
import logging
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch import distributed as dist
from torch.utils.data import (DataLoader, SequentialSampler)
from tqdm import tqdm
import datetime
from general_util.logger import setting_logger
from general_util.training_utils import set_seed, batch_to_device, unwrap_model
from general_util.mrr import get_mrr

from general_util.metrics import ROC, BA, RECALL

logger: logging.Logger

torch.multiprocessing.set_sharing_strategy('file_system')


def metrics(outputs, res):
    metric_items = ["auc", "ba", "recall@10"]
    for m in metric_items:
        if not res.__contains__(m):
            res[m] = 0
        res[m] += outputs[m]
    return res

def evaluate(cfg, model, embedding_memory=None, prefix="", _split="val"):
    dataset, collator = load_and_cache_examples(cfg, embedding_memory=embedding_memory, _split=_split)

    cfg.eval_batch_size = cfg.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(dataset)  # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=cfg.eval_batch_size, collate_fn=collator,
                                 num_workers=cfg.eval_num_workers if hasattr(cfg, "eval_num_workers"
                                                                             ) and cfg.eval_num_workers else cfg.num_workers)
    single_model_gpu = unwrap_model(model)
    # single_model_gpu.get_eval_log(reset=True)
    # Eval!
    # torch.cuda.empty_cache()
    logger.info("***** Running evaluation {}.{} *****".format(_split, prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", cfg.eval_batch_size)
    # Seems FSDP does not need to unwrap the model for evaluating.
    model.eval()

    res = {}

    count = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch['epoch'] = torch.tensor(0)
        batch = batch_to_device(batch, cfg.device)
        count += 1
        with torch.no_grad():
            if cfg.fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            for k, v in outputs.items():
                if not res.__contains__(k):
                    res[k] = []
                res[k].append(v)

    logger.info("****** Evaluation Results ******")
    logger.info(f"Global Steps: {prefix}")
    # logger.info(f"****** MRR: {str(mrr)} *********")
    # metric_log, results = single_model_gpu.get_eval_log(reset=True)
    
    logger.info("-----------------------------")
    for k, v in res.items():
        v = torch.cat(v, dim =0)
        res[k] = v
        logger.info(f"{k}: {v.size()}")

    return res


def load_and_cache_examples(cfg, embedding_memory=None, _split="test", _file=None):
    if cfg.local_rank not in [-1, 0] and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if _file is not None:
        input_file = _file
    elif _split == "train":
        input_file = cfg.train_file
    elif _split == "val":
        input_file = cfg.val_file
    elif _split == "test":
        input_file = cfg.test_file
    else:
        raise RuntimeError(_split)

    dataset = hydra.utils.instantiate(cfg.dataset, triple_file=input_file, split='test', embedding=embedding_memory)
    if hasattr(cfg, "collator"):
        if embedding_memory is not None:
            collator = hydra.utils.instantiate(cfg.collator, embedding=embedding_memory)
        else:
            collator = hydra.utils.instantiate(cfg.collator)
    else:
        collator = None

    if cfg.local_rank == 0 and _split == "train":
        dist.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return dataset, collator


@hydra.main(config_path="config", config_name="3_graph_FOTA_predict")
def main(cfg: DictConfig):
    global device
    if cfg.local_rank == -1 or cfg.no_cuda:
        device = str(torch.device("cuda" if torch.cuda.is_available() and not cfg.no_cuda else "cpu"))
        cfg.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(cfg.local_rank)
        device = str(torch.device("cuda", cfg.local_rank))
        dist.init_process_group(backend='nccl')
        cfg.n_gpu = 1
    cfg.device = device

    global _output_dir
    _output_dir = cfg.output_dir
    startdate = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    os.makedirs(_output_dir, exist_ok=True)
    _output_dir = os.path.join(_output_dir, startdate)
    os.makedirs(_output_dir, exist_ok=True)

    global logger
    logger = setting_logger(_output_dir, local_rank=cfg.local_rank)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   cfg.local_rank, device, cfg.n_gpu, bool(cfg.local_rank != -1), cfg.fp16)

    # Set seed
    set_seed(cfg)

    embedding_memory = hydra.utils.instantiate(cfg.embedding_memory) if hasattr(cfg, "embedding_memory") else None

    # Test
    results = {}
    checkpoints = [cfg.output_dir]
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        split = "val"

        state_dict = torch.load(os.path.join(checkpoint, "pytorch_model.bin"))
        model: torch.nn.Module = hydra.utils.call(cfg.model)
        model.load_state_dict(state_dict)
        model.to(device)

        if cfg.test_file:
            prefix = 'test-' + prefix
            split = "test"

        result = evaluate(cfg, model, embedding_memory=embedding_memory, prefix=prefix, _split=split)
        
        save_pt(cfg.model.user_vocab, result["user_emb"], result["user_index"], os.path.join(_output_dir, "user_emb_graph.pt"))
        save_pt(cfg.model.item_vocab, result["item_emb"], result["item_index"], os.path.join(_output_dir, "item_emb_graph.pt"))
        save_pt(cfg.collator.scene_vocab, result["scene_emb"], result["scene_index"], os.path.join(_output_dir, "scene_emb_graph.pt"))
        save_pt(cfg.collator.outfit_vocab, result["outfit_emb"], result["outfit_index"], os.path.join(_output_dir, "outfit_emb_graph.pt"))
            
    return results

def save_pt(vocab, emb, index, file):
    vocab = json.load(open(vocab))
    emb_arr = torch.zeros((len(vocab),emb.size()[1]))
    for i, idx in enumerate(index):
        emb_arr[idx, ] = emb[i,]
    
    torch.save(emb_arr, file)
    logger.info(f"Emb saved to {file}, {emb_arr.size()}")

if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--"):])
        else:
            hydra_formatted_args.append(arg)
    sys.argv = hydra_formatted_args

    main()

