from multiprocessing.sharedctypes import Value
from typing import List, Callable, Optional
import torch
import torch.utils.data as tdata
from functools import cached_property
from PIL import Image
import os.path as osp

from hake_utils.model_zoo import clip


class BaseVideoRecord(object):
    """
    Represents a video segment with an associated label.
    """
    def __init__(self, start: int, stop: int, path_gen: Callable[[int], str], file_check: bool=False):
        """video clip in frame range [start, stop]"""
        if start > stop:
            raise ValueError(f"Frame range error: {start} > {stop}")
        if file_check:
            if not osp.exists(path_gen(start)):
                raise ValueError(f"File not exists {path_gen(start)}")
            if not osp.exists(path_gen(stop)):
                raise ValueError(f"File not exists {path_gen(stop)}")

        self.start_frame = start
        self.stop_frame = stop
        self.gen_image_path = path_gen

    @cached_property
    def num_frames(self) -> int:
        return self.stop_frame - self.start_frame + 1

    def sample_video_segment(self, frame_sampler) -> List[Image.Image]:
        """Sample a video clip with given size.
        Return full video if segment_size is not set."""
        frame_ids = frame_sampler.sample(self.start_frame, self.stop_frame+1)
        
        images = []
        for index in frame_ids:
            images.append(Image.open(
                osp.join(self.gen_image_path(index))
            ).convert("RGB"))
        
        return images


class BaseDataset(tdata.Dataset):
    """
    Interfaces for users:
    @ methods
    
        __init__(root: str, 
                split="train"|"test", 
                resolution: int,
                task="verb"|"hoi")
                            
        __len__() -> number of instance (not image)
        
        __getitem__(i) -> {
            'images': tensor of [n_frame, 3, resolution, resolution]
            'prompt': str
            'labels': int(single-label) / tensor[num_class](multi-label)
        }

    @ properties
        prompt_text_per_class
        prompt_token_per_class

    
    Protocols for developers:
    @ to override
        __init__(...)

    @ abstract methods:
        __len__()
        __getitem__(i)
        prompt_text_per_class()
    """

    # - instance level的索引（因为这里不需要涉及多种）
    # - image和video都用video的格式返回
    # - single-label和multi-label一样的返回list of label


    def __init__(self, root: str, split: str, task: str):
        super().__init__()

        self.root = root
        self.split = split
        self.task = task

    def __getitem__(self, index) -> dict:
        raise NotImplementedError("__getitem__")
    
    @cached_property
    def prompt_text_per_class(self) -> List[List[str]]:
        raise NotImplementedError("prompt_text_per_class")

    @cached_property
    def prompt_token_per_class(self) -> List[torch.Tensor]:
        all_tokens = []
        for prompts in self.prompt_text_per_class:
            all_tokens.append(clip.tokenize(prompts))
        return all_tokens
