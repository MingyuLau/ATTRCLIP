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
from .prompter.Breakfast import Prompter as Prompter
from models import clipmvit_model

import utils.io as hake_io

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

        if self.frames > 1:
            # adjust stride smaller to make the "span" of the segment smaller than clip size
            max_stride = (clip_size - 1)*1.0 / \
                (self.frames - 1)  # support float stride
            if max_stride < stride:
                stride = max_stride

        segment_span = int((self.frames-1)*stride) + 1

        offset = np.random.randint(clip_size - segment_span + 1)
        res = np.linspace(range_begin+offset, range_begin +
                          offset+segment_span-1, self.frames).tolist()
        return [int(x) for x in res]


class BreakfastVideoRecord(object):
    """
    Represents a video segment with an associated label.
    """

    def __init__(self, data_root: str, annot_row: Dict[str, Any], narration: str, narration_id: int):
        participant_id = cast(str, annot_row["participant_id"])
        type_id = cast(str, annot_row["type_id"])
        video_id = cast(str, annot_row['video_id'])
        frame_root = osp.join(data_root, "frame", participant_id, type_id, video_id)

        self.verb_id = narration_id
        self.verb_str = narration
        self.start_frame = int(annot_row["start_frame"])
        self.stop_frame = int(annot_row["end_frame"])
        self.template = osp.join(frame_root, "frame_{:010d}.jpg")

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


@DATASET_REGISTRY.register()
class Breakfast(torch.utils.data.Dataset):
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

        # list of classes
        self.verb2id_list = hake_io.load_json(root, "narration_list.json")
        self.verb_list = [None]*len(self.verb2id_list)
        for k, i in self.verb2id_list.items():
            self.verb_list[i] = k
        logger.info(
            f"{len(self.verb2id_list)} verbs")

        # all annotation
        annot_file = {
            "train": "train_annotation_new.csv",
            "val": "validation_annotation_new.csv",
        }[mode]

        self.video_records = []
        with open(osp.join(root, annot_file), "r") as fp:
            for row in csv.DictReader(fp):
                # frame of doing nothing, don't append
                narration = row["narration"]
                narration_id = int(self.verb2id_list[narration])
                
                if narration == 'SIL':
                    continue
                self.video_records.append(BreakfastVideoRecord(root, row, narration, narration_id))
        logger.info(f"{mode} = {len(self.video_records)} video clips")

        # visual and linguistic preprocesser
        self.transform = hake_transforms.default_transform_EPIC(
            "train" if mode == "train" else "valid",
            cfg.DATA.TRAIN_CROP_SIZE if mode == "train" else cfg.DATA.TEST_CROP_SIZE
        )
        self.frame_sampler = FrameSampler(cfg.DATA.NUM_FRAMES, cfg.DATA.FPS)
        self.prompter = Prompter()

        self.repeat_time = 1 if mode == "train" else cfg.DATA.NUM_EVAL_VIEWS

    # def __read_class_list_file(self, filepath: str) -> List[str]:
    #     class_list = []
    #     with open(osp.join(self.root, filepath), "r") as fp:
    #         for row in csv.DictReader(fp):
    #             assert int(row['id']) == len(
    #                 class_list), "There is a missing id"
    #             class_list.append(row['key'])
    #     return class_list

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
            return images + [record.verb_id, prompt]

        else:
            return images + [record.verb_id]
            # frm, ch, r, r

    @cached_property
    def prompt_token_per_class(self) -> List[torch.Tensor]:
        all_prompts = []
        for verb_str in self.verb2id_list:
            all_prompts.append(self.prompter.list_all(verb_str))

        all_tokens = []
        for prompts in all_prompts:
            all_tokens.append(clipmvit_model.tokenize(prompts))
        return all_tokens
