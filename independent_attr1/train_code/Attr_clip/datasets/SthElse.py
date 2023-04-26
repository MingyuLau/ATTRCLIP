# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import csv
from functools import cached_property, lru_cache
import gc
import json
import pickle
import numpy as np
import os.path as osp
from typing import List

import torch
import torch.utils.data
from PIL import Image

from .build import DATASET_REGISTRY
import utils.logging as logging

import hake_utils.data.transforms as hake_transforms
from .prompter.EgoHand import Prompter as Prompter
from models import clipmvit_model

import hake_utils.io as hake_io

logger = logging.get_logger(__name__)



class FrameSampler(object):
    def __init__(self, frames, stride):
        """Sample a subset of frames with maximum size
        frames: number of frames
        stride: the gap between two sampled frames"""

        self.frames = frames
        self.stride = stride

    def sample(self, range_begin: int, range_end: int) -> List[int]: 
        """Sample a video clip with given size."""

        clip_size = range_end - range_begin
        stride = self.stride

        if self.frames>1:
            # adjust stride smaller to make the "span" of the segment smaller than clip size
            max_stride = (clip_size - 1)*1.0 / (self.frames - 1)  # support float stride
            if max_stride < stride:
                stride = max_stride


        segment_span = int((self.frames-1)*stride) + 1

        offset = np.random.randint(clip_size - segment_span + 1)
        res = np.linspace(range_begin+offset, range_begin+offset+segment_span-1, self.frames).tolist()
        return [int(x) for x in res]


@lru_cache(maxsize=None)
def load_bounding_box_smthsmth(data_root: str):
    # return hake_io.load_pickle_bin(data_root, "bounding_box_smthsmth_merge.pkl")
    return hake_io.load_json(data_root, "sthelse_videos.json")

@lru_cache(maxsize=None)
def frame_number(data_root: str):
    return hake_io.load_json(data_root, "frame_number.json")

        
class SthElseVideoRecord(object):
    """
    Represents a video segment with an associated label.
    """
    def __init__(self, data_root: str, video_id: str,
                    template: str, verb_str: str, verb_id: int, box_info):
        self.template = template
        self.verb_str = verb_str
        self.verb_id = verb_id
        self.box_info = box_info

        image_path_root = osp.join(data_root, "frame", video_id)

        self.start_frame = 0
        self.stop_frame = frame_number(data_root)[video_id]-1

        self.path_template = osp.join(image_path_root, "{:010d}.jpg")

    @cached_property
    def num_frames(self) -> int:
        return self.stop_frame - self.start_frame + 1

    def sample_video_segment(self, frame_sampler: FrameSampler) -> List[Image.Image]:
        """Sample a video clip with given size.
        Return full video if segment_size is not set."""
        frame_ids = frame_sampler.sample(self.start_frame, self.stop_frame+1)
        
        images = []
        for index in frame_ids:
            try:
                images.append(Image.open(
                    self.path_template.format(index)
                ).convert("RGB"))
            except OSError as e:
                raise OSError(self.path_template.format(index))
                
        
        return images



@DATASET_REGISTRY.register()
class SthElse(torch.utils.data.Dataset):
    """
    iteration returns {
        images: Tensor[n_frame, 3, resol., resol.]
        prompts: str (only during training)
        labels: int
        # frame_hoa       HOA[n_frame]
        # verb_name       string
        # verb_id         int
        # obj_name_list   string[]
        # obj_id_list     int[]
    }
    """

    def __init__(self, cfg, mode, num_retries=10):
        self.root = root = cfg.DATA.PATH_TO_DATA_DIR
        self.mode = mode
        self.task = cfg.TASK

        with open(osp.join(root, "SSv2/something-something-v2-labels.json")) as fp:
            self.verb2index_dict = json.load(fp)
        self.verb2index_dict = {k.lower():int(v) for k, v in self.verb2index_dict.items()}
        rev_verb_list = {v: k for k, v in self.verb2index_dict.items()}
        self.verb_list = [rev_verb_list[i] for i in range(len(rev_verb_list))]
        
        logger.info(f"{len(self.verb_list)} verbs")
            
        # annotations
        logger.info(f"Loading SthElse annotation")
        sthelse_annot = load_bounding_box_smthsmth(root)
        sthelse_annot = set(sthelse_annot) # id->annot
        # [{'name': '114040/0001.jpg',
        #   'labels': [{'box2d': [183.20441988950276, 348.0662983425414, 81.32596685082872, 144.5303867403315],
        #     'category': 'highlighter',
        #     'gt_annotation': 'object 0',
        #     'standard_category': '0000'},
        #    {'box2d': [0, 137.23756906077347, 158.232044198895, 240],
        #     'category': 'hand',
        #     'gt_annotation': 'object hand',
        #     'standard_category': 'hand'}],
        #   'gt_placeholders': ['highlighter'],
        #   'nr_instances': 2},
        #  {'name': '114040/0002.jpg',
        #   'labels': [{'box2d': [182.5544361390965, 347.41631459213517, 80.62398440038999, 143.82840428989277],
        #     'category': 'highlighter',
        #     'gt_annotation': 'object 0',
        #     'standard_category': '0000'},
        #    {'box2d': [0, 169.06077348066296, 129.94475138121547, 240],
        #     'category': 'hand',
        #     'gt_annotation': 'object hand',
        #     'standard_category': 'hand'}],
        #   'gt_placeholders': ['highlighter'],
        #   'nr_instances': 2},
        #  ...

        logger.info(f"Loading SthSth annotation")
        sthsth_annot_file = {
            "train": "something-something-v2-train.json",
            "val": "something-something-v2-validation.json",
        }[mode]
        sthsth_annot = hake_io.load_json(root, "SSv2", sthsth_annot_file)
        # {   "id":"74225",
        #     "label":"spinning cube that quickly stops spinning",
        #     "template":"Spinning [something] that quickly stops spinning",
        #     "placeholders":["cube"]   }

        # merge annotations from sthsth and sthelse
        self.video_records = []
        for sample in sthsth_annot:
            if sample["id"] in sthelse_annot:
                video_id = sample['id']
                template = sample['template'].replace("[something]", "{}")
                verb_str = sample['template'].replace("[", "").replace("]", "").lower()
                verb_id  = self.verb2index_dict[verb_str]
                record = SthElseVideoRecord(
                    osp.join(root, "SSv2"), video_id, 
                    template, verb_str, verb_id,
                    None
                    # sthelse_annot[sample["id"]]
                )
                self.video_records.append(record)
        
        # del sthelse_annot
        # gc.collect()
        

        # visual and linguistic preprocesser
        self.transform = hake_transforms.default_transform_EPIC(
            "train" if mode=="train" else "valid",
            cfg.DATA.TRAIN_CROP_SIZE if mode=="train" else cfg.DATA.TEST_CROP_SIZE
        )
        self.frame_sampler = FrameSampler(cfg.DATA.NUM_FRAMES, cfg.DATA.FPS)
        self.prompter = Prompter()

        self.repeat_time =  1




    def __len__(self) -> int:
        return len(self.video_records)

    def __getitem__(self, index):

        images = []

        record = self.video_records[index]
        frame_images = record.sample_video_segment(self.frame_sampler)
        frame_transformed = self.transform(frame_images)
        # frame_transformed = torch.stack(frame_transformed, 0)
        del frame_images
        images.append(frame_transformed)


        if self.mode == "train":
            prompt = self.prompter.random_choice(record.verb_str)
            prompt = clipmvit_model.tokenize(prompt).squeeze(0)
            return images + [record.verb_id, prompt]
            
        else:
            return images + [record.verb_id]
            # frm, ch, r, r
        

    @cached_property
    def prompt_token_per_class(self) -> List[torch.Tensor]:
        all_prompts = []
        for verb_str in self.verb_list:
            all_prompts.append(self.prompter.list_all(verb_str))

        all_tokens = []
        for prompts in all_prompts:
            all_tokens.append(clipmvit_model.tokenize(prompts))
        return all_tokens
