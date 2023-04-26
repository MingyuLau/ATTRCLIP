# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

import csv
from functools import cached_property
import numpy as np
import os.path as osp
from typing import Any, Dict, List, cast

import torch
import torch.utils.data
from PIL import Image

from .build import DATASET_REGISTRY
import utils.logging as logging

import hake_utils.data.transforms as hake_transforms
from .prompter.EgoHand import Prompter as Prompter
from models import clipmvit_model

logger = logging.get_logger(__name__)


# 这是一个针对Epic-Kitchens-100数据集的pytorch数据集实现，该数据集类使用装饰器@DATASET_REGISTRY.register()注册到
# DATASET_REGISTRY对象中华，构造函数接受配置对象cfg，数据集模式mode和一个参数num_retries, 它设置在视频剪辑加载失败
# 时重试的次数

# 内部类FrameSampler，它从视频剪辑中采样一部分帧。它接受帧数和步长作为输入，并返回要采样的帧的索引列表
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


#类 EpicVideoRecord 表示带有关联标签的视频段。它接受数据集的根目录和包含视频剪辑注释的字典作为输入。它提取有关视频剪辑的必要信息，例如开始和结束帧、动词 ID 和帧模板。        
class EpicVideoRecord(object):
    """
    Represents a video segment with an associated label.
    """
    def __init__(self, data_root: str, annot_row: Dict[str, Any]):
        participant_id = cast(str, annot_row["participant_id"])
        video_id = cast(str, annot_row['video_id'])
        annot_root = osp.join(data_root, "EPIC-KITCHENS", participant_id)

        self.verb_id = int(annot_row["verb_class"]) # int
        self.verb_str = annot_row["verb"] # str
        # self.noun_id = int(annot_row["noun_class"])  # int
        # self.noun_str = annot_row["noun"]  # str

        self.start_frame = int(annot_row["start_frame"])
        self.stop_frame = int(annot_row["stop_frame"])

        self.template = osp.join(annot_root, "rgb_frames", video_id, "frame_{:010d}.jpg")

    @cached_property
    def num_frames(self) -> int:
        return self.stop_frame - self.start_frame + 1

    def sample_video_segment(self, frame_sampler: FrameSampler) -> List[Image.Image]:
        """Sample a video clip with given size.
        Return full video if segment_size is not set."""
        frame_ids = frame_sampler.sample(self.start_frame, self.stop_frame+1)
        
        images = []
        for index in frame_ids:
            images.append(Image.open(
                self.template.format(index)
            ).convert("RGB"))
        
        return images


# 该函数将被自动注册到'DATASET_REGISTRY'对象中，以其名称作为注册表的键，
# 该数据类有以下属性：
# root：数据集的根目录
# mode：数据集的模式，可以是train或者val
# task：使用数据集的任务
# verb_list: 所有动词类别的列表
# noun_list: 所有名词类别的列表
# video: 视频剪辑记录的列表
# frame_sampler: 从视频剪辑中采样帧的FrameSampler对象
# num_retries: 在视频剪辑加载失败时重试的次数
@DATASET_REGISTRY.register()
class Epic(torch.utils.data.Dataset):
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
    # def __init__(self, root: str, split: str, task: str, data_cfg: ConfigManager):
        # super().__init__(root, split, task)

        self.root = root = cfg.DATA.PATH_TO_DATA_DIR
        self.mode = mode
        self.task = cfg.TASK

        # list of classes 加载了数据集中的动词和名词列表，但代码中似乎并没有用到名词列表
        self.verb_list = self.__read_class_list_file(
            "epic-kitchens-100-annotations/EPIC_100_verb_classes.csv")
        self.noun_list = self.__read_class_list_file(
            "epic-kitchens-100-annotations/EPIC_100_noun_classes.csv")
        logger.info(f"{len(self.verb_list)} verbs, {len(self.noun_list)} nouns")
            
        # all annotation
        annot_file = {
            "train": "epic-kitchens-100-annotations/EPIC_100_train.csv",
            "val": "epic-kitchens-100-annotations/EPIC_100_validation.csv",
        }[mode]
        
        self.video_records = []
        with open(osp.join(root, annot_file), "r") as fp:
            for row in csv.DictReader(fp):
                self.video_records.append(EpicVideoRecord(root, row))
        logger.info(f"{mode} = {len(self.video_records)} video clips")

        # visual and linguistic preprocesser
        self.transform = hake_transforms.default_transform_EPIC(
            "train" if mode=="train" else "valid",
            cfg.DATA.TRAIN_CROP_SIZE if mode=="train" else cfg.DATA.TEST_CROP_SIZE
        )
        self.frame_sampler = FrameSampler(cfg.DATA.NUM_FRAMES, cfg.DATA.FPS)
        self.prompter = Prompter()

        self.repeat_time = 1 if mode == "train" else cfg.DATA.NUM_EVAL_VIEWS



    def __read_class_list_file(self, filepath: str) -> List[str]:
        class_list = []
        with open(osp.join(self.root, filepath), "r") as fp:
            for row in csv.DictReader(fp):
                assert int(row['id'])==len(class_list), "There is a missing id"
                class_list.append(row['key'])
        return class_list

    def __len__(self) -> int:
        return len(self.video_records)

    def __getitem__(self, index):

        images = []

        for _ in range(self.repeat_time):
            record = self.video_records[index]
            frame_images = record.sample_video_segment(self.frame_sampler)
            frame_transformed = self.transform(frame_images)
            # frame_transformed = torch.stack(frame_transformed, 0)
            del frame_images
            images.append(frame_transformed)


        if self.mode == "train":
            prompt = self.prompter.random_choice(record.verb_str)
            prompt = clipmvit_model.tokenize(prompt).squeeze(0)
            return images + [record.verb_id, prompt] # test才会用到
        # 两个列表相加，合并后将得到一个新的列表，因此最终返回的列表包含了视频帧数据，动词标签和动词注释
    
        else:
            return images + [record.verb_id]        # 这里的verb_id是什么
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
