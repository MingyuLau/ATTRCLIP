#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

from fvcore.common.registry import Registry

# 使用fvcore.common.registry定义了一个名为DATASET_REGISTRY的注册表，用于存储数据集构建函数，
# 使用DATASET_REGISTRY.register()方法来将不同的数据集构建函数添加到注册表中，当需要创建特定的
# 数据集时，可以使用DATASET_REGISTRY.get()方法来获取相应的构建函数，并将配置和数据集划分作为参数传递给改函数来
# 创建数据集对象。
DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for dataset.

The registered object will be called with `obj(cfg, split)`.
The call should return a `torch.utils.data.Dataset` object.
"""


def build_dataset(dataset_name, cfg, split):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        dataset_name (str): the name of the dataset to be constructed.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # Capitalize the the first letter of the dataset_name since the dataset_name
    # in configs may be in lowercase but the name of dataset class should always
    # start with an uppercase letter.
    name = dataset_name
    if 'a' <= name[0] <= 'z':
        name = dataset_name.capitalize()     # 将首字母大写

    

    return DATASET_REGISTRY.get(name)(cfg, split)

# DATASET_REGISTRY.get(name)(cfg, split)从注册表中获取‘name'对应的构建函数，并将’cfg‘和’split‘作为参数传递给
# 该函数，从而创建数据集对象
