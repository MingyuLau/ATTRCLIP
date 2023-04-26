#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.

"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for models.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in mvit/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)  # 这段代码从全局变量MODEL_REGISTRY中获取一个名为name的模型构建函数，并将配置参数cfg作为参数传递给改构建函数，从而构建出一个pytorch模型对象model

    if len(cfg.MODEL.FREEZE_PARAM) > 0:
        pass
        

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        # 将model包装在xxx中，从而实现在多个GPU上进行并行训练，具体来说，xxx将模型的参数划分为
        # 并将每个部分分配给不同的gpu设备进行计算。最后将结果胡总
        # module: 要包装的模型对象
        # device_ids: 用于进行并行训练的gpu设备的id列表
        # output_device: 指定输出结果的gpu设备的id

    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("Model is built, params =", pytorch_total_params//1000000, "M")

    return model
