import csv
import pdb
import json
import torch
import numpy as np
from PIL import Image
import os.path as osp
import torch.utils.data
import sys
sys.path.append('/home/user/lmy/DATA/independent_attr1/train_code/abcd/datasets/my_transforms.py')
from utils import logging
from functools import cached_property
from typing import Any, Dict, List, cast
import utils.my_transforms as attr_transforms
from .prompter.attr import Prompter as Prompter
from models import clipimg_model  
from .build import DATASET_REGISTRY


# logger = logging.get_logger(__name__)



@DATASET_REGISTRY.register()
class CGQA(torch.utils.data.Dataset):
    """
    iteration returns {
        images: Tensor
        prompts: str (only during traning)
        labels: int
    }
    """

    def __init__(self, cfg, mode):

        self.root = '/home/user/lmy/DATA/independent_attr1/train_code/abcd/data/C-GQA/images/'
        self.mode = mode 
        self.task = cfg.TASK
        self.img_datas = []
        self.attr_prompt_pair = []
        self.pair = {}
        self.attr_list = []
        raw_dataset = torch.load('/home/user/lmy/DATA/independent_attr1/train_code/abcd/data/C-GQA/metadata_compositional-split-natural.t7')
        
        self.verb_list = []
        with open('/home/user/lmy/DATA/independent_attr1/train_code/abcd/data/C-GQA/CGQA.txt', 'r') as f:
            for line in f.readlines():
                self.verb_list.append(line.strip())
        # pdb.set_trace()
        self.prompter = Prompter()
        for idx, data in enumerate(raw_dataset):
            if data['set'] == 'test':
                self.img_datas.append(data)
        # pdb.set_trace()
        with open('/home/user/lmy/DATA/independent_attr1/train_code/abcd/data/C-GQA/semantic_attr_train.json', 'r') as f:
            anno = json.load(f)
            for idx, item in enumerate(anno['pair']):
                self.pair[item[1]] = item[2] 
                self.attr_list.append(item[1])
        self.attr_prompt_pair = anno['pair']     # [['CGQA', 'calm', 'If an object is calm, it may appear to be still or at rest.']]
        for i in self.attr_prompt_pair:
            self.pair[i[1]] = i[2]
        #pdb.set_trace()
        self.transform = attr_transforms.default_transform_CLIP(224)
        # self.repeat_time = 1 if mode == "train" else cfg.DATA.NUM_EVAL_VIEWS
        
            
        self.repeat_time = 1
    def __len__(self) -> int:
        return len(self.img_datas) 
    def __getitem__(self, index):
        
        img_data = self.img_datas[index]
        # pdb.set_trace()
        attr_id = self.verb_list.index(img_data['attr'])
        image = Image.open(self.root + img_data['image']).convert('RGB')
        prompt = self.pair[img_data['attr']]
        prompt = clipimg_model.tokenize(prompt).squeeze(0)
        #pdb.set_trace()
        image = self.transform(image)
        
        if self.mode == "train":
            return [image, attr_id, prompt]
        else:
            return [image, attr_id]
    
    @cached_property
    def prompt_token_per_class(self) -> List[torch.Tensor]:
        all_prompts = []
        for verb_str in self.verb_list:
            # all_prompts.append(self.prompter.list_all(verb_str))
            all_prompts.append([self.pair[verb_str]])
            # all_prompts.append([verb_str])
        # pdb.set_trace()
        all_tokens = []
        for prompts in all_prompts:
            all_tokens.append(clipimg_model.tokenize(prompts))
        # pdb.set_trace()
        return all_tokens          # [413],动词列表和prompt列表一一对应，prompt列表是由列表组成的列表，每个列表是一个动词对应的所有prompt,all_tokens类似


# if __name__ == "__main__":
#     dset = CGQA('/home/user/lmy/DATA/independent_attr1/train_code/attr_clip/configs/123.yaml', "train")
#     dset[1]  
