from collections import Counter
from functools import cached_property
import json
import random
from typing import List
import torch
import torch.utils.data as tdata

from datasets.build import DATASET_REGISTRY, build_dataset
from .prompter.EgoHand import Prompter
from models import clipmvit_model


EXTRA_DIR = {
    "Epic": "data/epic-kitchens",
    "FiftySalads": "data/50salad",
    "Breakfast": "data/breakfast",
    "IKEA": "data/ikea_asm_dataset_public",
    "EgteaGaze": "data/EGTEA_GAZE+",
    "CharadesEgo": "data/charades-ego",
    "Ego4dAction": "data/Ego4d/ego4d-fho/",
    "Ego4dActionNew": "data/Ego4d/ego4d-fho/",
    "SthElse": "data/something-else/",
}



def purify_verb(x):
    x = x.strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return x



@DATASET_REGISTRY.register()
class One4AllDataset(tdata.Dataset):
    def __init__(self, cfg, mode):
        
        self.datasets = {}

        # train:xxx; val:xxx
        data_mask = cfg.TRAIN.EXTRA_MASK.split(";")
        data_mask = [x.split(":") for x in data_mask]
        mask_per_ds = {k.strip():v.strip() for k, v in data_mask} # mask_per_ds: {"train": "xxx", "val": "xxx"}
        mask_path = mask_per_ds[mode]                             # mask_path: "xxx"

        with open(mask_path, "r") as fp:
            self.idx_list = json.load(fp)


        idx_per_dataset = Counter([k for k, _ in self.idx_list]) # idx_per_dataset: {"Epic": 100, "FiftySalads": 200, ...}

        for name in idx_per_dataset.keys():
            cfg.DATA.PATH_TO_DATA_DIR = EXTRA_DIR[name] # cfg.DATA.PATH_TO_DATA_DIR: "data/epic-kitchens"
            dset = build_dataset(name, cfg, mode)       # dset: EpicDataset 
            self.datasets[name] = dset                  # self.datasets: {"Epic": EpicDataset, "FiftySalads": FiftySaladsDataset, ...}

        print(f"One4All {mode}")
        for k, v in idx_per_dataset.items():
            print(f"\t{k}: {v}")                        # print: "Epic: 100", "FiftySalads: 200", ...
 

        # verb-classes stuff
        self.verb2id_o4a = json.load(open("video_property/ego-o4a/verb2id_all.json")) # self.verb2id_o4a: {"verb1": 0, "verb2": 1, ...}

        self.num_class = len(set(self.verb2id_o4a.values()))
        self.verb_str_per_id = [[] for _ in range(self.num_class)]
        for k, i in self.verb2id_o4a.items():
            self.verb_str_per_id[i].append(k)

        self.prompter = Prompter()


    
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        name, idx = self.idx_list[index] # self.idx_list: [("Epic", 0), ("Epic", 1), ("FiftySalads", 0), ("FiftySalads", 1), ...]
        dset = self.datasets[name]
        item = dset[idx]                 # item: {"images": Tensor, "verb": "verb1", "verb_id": 0, "repeat_time": 1, "verb_str": "verb1", "verb_str_id": 0, "verb_str_id_o4a": 0, "verb_str_id_o4a_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,

        verb_id = item[dset.repeat_time]
        try:
            verb_str = dset.verb_list[verb_id]
        except KeyError:
            print(dset.verb_list, verb_id)
        
        verb_str = purify_verb(verb_str)
        item[dset.repeat_time] = self.verb2id_o4a[verb_str]

        return item



    @cached_property
    def prompt_token_per_class(self) -> List[torch.Tensor]:
        all_prompts = []
        
        for verb_str_list in self.verb_str_per_id:
            prompts = []
            for verb_str in verb_str_list:
                prompts += self.prompter.list_all(verb_str)
            all_prompts.append(prompts)

        all_tokens = []
        for prompts in all_prompts:
            all_tokens.append(clipmvit_model.tokenize(prompts))
        return all_tokens


# 代码功能
# 1. 读取多个不同的数据集，比如Epic, FiftySalads, Breakfast, IKEA, EgteaGaze, CharadesEgo, Ego4dAction, Ego4dActionNew, SthElse，并将他们合成一个更大的数据集
# 2. 将数据集中的动词标签转换为跟一般的动作标签
# 3. 生成每个动作标签的提示词prompt并转换为pytorch tensor

# 代码通过读取配置参数TRAIN.EXTRA_MASK中的信息来确定应该读取哪些数据集以及他们的位置
# 例如TRAIN.EXTRA_MASK = "train:video_property/ego-o4a/train.json; val:video_property/ego-o4a/val.json"

# prompt_token_per_class 属性是一个 cached_property，它用于生成提示语（prompt）对应的 PyTorch tensor。在该方法中，针对每种动作标签，根据预定义的模板生成多个提示语，并将这些提示语转换为 PyTorch tensor，最终返回一个列表，其中每个元素对应一种动作标签。