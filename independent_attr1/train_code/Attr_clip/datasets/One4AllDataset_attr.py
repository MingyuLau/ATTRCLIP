from collections import Counter
from functools import cached_property
import os
import sys
import pdb
import json
import random
from typing import List
import torch
import torch.utils.data as tdata
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.build import DATASET_REGISTRY, build_dataset
from .prompter.attr import Prompter as Prompter
from models import clipimg_model


EXTRA_DIR = {
    # "CGQA": "data/C-GQA",
    "VG": "data/VG",
}



def purify_verb(x):
    x = x.strip().lower()
    x = x.replace("_", " ").replace("-", " ")
    return x



@DATASET_REGISTRY.register()
class One4AllDataset_Attr(tdata.Dataset):
    def __init__(self, cfg, mode):
        
        self.datasets = {}   
        cfg.TRAIN.EXTRA_MASK = "train:/home/user/lmy/DATA/independent_attr1/train_code/abcd/video_property/One4all_train_100K_sc5.json; val:/home/user/lmy/DATA/independent_attr1/train_code/abcd/video_property/One4all_train_100K_sc5.json"   
        # cfg_TRAIN_EXTRA_MASK = "train:/home/user/lmy/DATA/independent_attr1/train_code/abcd/configs/One4all_train_10K_sc5.json; val:video_property/ego-o4a/val.json"
        # train:xxx; val:xxx
        data_mask = cfg.TRAIN.EXTRA_MASK.split(";")               # cfg.TRAIN.EXTRA_MASK: "train:xxx; val:xxx"
        data_mask = [x.split(":") for x in data_mask]             # data_mask: [["train", "xxx"], ["val", "xxx"]]
        mask_per_ds = {k.strip():v.strip() for k, v in data_mask} # mask_per_ds: {"train": "xxx", "val": "xxx"}
        mask_path = mask_per_ds[mode]                             # mask_path: "xxx"

        with open(mask_path, "r") as fp:
            self.idx_list = json.load(fp)      # self.idx_list: [["CGQA", 0], ["CGQA": 1], ...]                      

        idx_per_dataset = Counter([k for k, _ in self.idx_list]) # idx_per_dataset: Counter({'CGQA': 5637, 'DTD': 4363})
        # pdb.set_trace()


        for name in idx_per_dataset.keys():
            
            #cfg.DATA.PATH_TO_DATA_DIR = EXTRA_DIR[name] 
            cfg_DATA_PATH_TO_DATA_DIR = EXTRA_DIR[name]
            # pdb.set_trace()
            dset = build_dataset(name, cfg, mode)       # dset: EpicDataset 
            self.datasets[name] = dset                  # self.datasets: {"Epic": EpicDataset, "FiftySalads": FiftySaladsDataset, ...}
        # pdb.set_trace()
        print(f"One4All {mode}")                        # print: "One4All train"
        for k, v in idx_per_dataset.items():
            print(f"\t{k}: {v}")                        # print: "Epic: 100", "FiftySalads: 200", ...
 

        # verb-classes stuff
        self.verb2id_o4a = json.load(open("/home/user/lmy/DATA/independent_attr1/train_code/abcd/attr2id_vg.json")) # self.verb2id_o4a: {"verb1": 0, "verb2": 1, ...}
        self.prompter = Prompter()
        self.num_class = len(set(self.verb2id_o4a.values()))
        self.verb_str_per_id = [[] for _ in range(self.num_class)]  # 这一句代码创建了一个包含 self.num_class 个空列表的列表 self.verb_str_per_id。
        
        for k, i in self.verb2id_o4a.items():
            #pdb.set_trace()
            self.verb_str_per_id[i].append(eval(k))     # 一个编号对应多个动词
        # pdb.set_trace()
        # pdb.set_trace()
        # self.prompter = Prompter()

    
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        name, idx = self.idx_list[index] # self.idx_list: [["CGQA", 20877], ["CGQA": 23], ...]
        dset = self.datasets[name]
        item = dset[idx]                 # 从子数据集中获取数据，具体参考每个数据集专用的dataset
        verb_id = item[dset.repeat_time]   
        try:
            verb_str = dset.attr_list[verb_id]
        except KeyError:
            print(dset.attr_list, verb_id)
        
        #verb_str = purify_verb(verb_str)
        
        item[dset.repeat_time] = self.verb2id_o4a[str(verb_str)] # item

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
            all_tokens.append(clipimg_model.tokenize(prompts))
        return all_tokens


if __name__ == "__main__":

    attr_dataset = One4AllDataset_Attr(None, "train")
    attr_dataset[1]

