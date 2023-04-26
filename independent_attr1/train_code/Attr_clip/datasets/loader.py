#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Data loader."""

import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from .build import build_dataset
# 这里是python的相对导入语法，在当前包(package)下引入其他模块或者包
# 在python3中，使用.表示当前包，..表示上一级包
# 原因是在python中，如果一个模块或者包没有在当前路径下，python解释器
# 会默认在系统路径中查找该模块或者包，使用相对导入可以避免模块名称冲突和命名空间污染的问题


# 用于在训练时对一个batch中的多个样本进行重复增强操作，并将增强后的样本进行扁平化flatten，返回扁平化后的输入和标签，代码里面似乎没用到
def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]

    inputs, labels = default_collate(inputs), default_collate(labels)
    # default_collate是python中的一个函数，用于将一个list或tuple类型的batch转化成一个tensor类型的batch，
    # 在数据加载时，通常需要架将多个样本组成一个batch，并对每个样本进行预处理或者增强操作，最后将处理后的样本组成一个
    # batch返回。由于不同样本的维度和类型可能不同，因此需要将他们转化为相同的tensor类型，并按照batch的第一维度进行堆叠



    return inputs, labels

# TODO：重点！根据配置文件和数据集划分构建对应的数据加载器
# 在训练集上，使用distributedSampler实现多进程并行读取数据，同时支持多次增强的批处理
# 在验证集和测试集上，使用普通的随机采样和顺序采样读取数据
def construct_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None

    # if cfg.AUG.NUM_SAMPLE > 1 and split in ["train"]:
    #     collate_func = multiple_samples_collate
    # else:
    #     collate_func = None
    collate_func = None

    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),   
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_func,     
    )
    return loader
    # 当enumerate loader的时候，enumerate函数返回的每个元素是一个二元组(batch_idx, batch_data)
    # batch_data是Dataloader对象返回的数据批次，通常是一个元组或者字典，包含输入数据和对应的标签

# 用于在训练时对数据集进行随机打乱，以增加模型的训练随机性和泛化性
def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the dataset.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)
