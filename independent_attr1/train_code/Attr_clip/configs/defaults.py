# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

"""Configs."""
import math
from fvcore.common.config import CfgNode
# 使用CfgNode类可以将配置参数组织成树形结构，每个节点都可以有多个子节点和属性。使得配置参数的管理和访问变得更加简单和直观
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


_C.CONTRASTIVE = CfgNode()

# temperature used for contrastive losses
_C.CONTRASTIVE.T = 0.07

# output dimension for the loss
_C.CONTRASTIVE.DIM = 128

# number of training samples (for kNN bank)
_C.CONTRASTIVE.LENGTH = 239975

# the length of MoCo's and MemBanks' queues
_C.CONTRASTIVE.QUEUE_LEN = 65536

# momentum for momentum encoder updates
_C.CONTRASTIVE.MOMENTUM = 0.5

# wether to anneal momentum to value above with cosine schedule
_C.CONTRASTIVE.MOMENTUM_ANNEALING = False

# either memorybank, moco, simclr, byol, swav
_C.CONTRASTIVE.TYPE = "mem"

# wether to interpolate memorybank in time
_C.CONTRASTIVE.INTERP_MEMORY = False

# 1d or 2d (+temporal) memory
_C.CONTRASTIVE.MEM_TYPE = "1d"

# number of classes for online kNN evaluation
_C.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM = 400

# use an MLP projection with these num layers
_C.CONTRASTIVE.NUM_MLP_LAYERS = 1

# dimension of projection and predictor MLPs
_C.CONTRASTIVE.MLP_DIM = 2048

# use BN in projection/prediction MLP
_C.CONTRASTIVE.BN_MLP = False

# use synchronized BN in projection/prediction MLP
_C.CONTRASTIVE.BN_SYNC_MLP = False

# shuffle BN only locally vs. across machines
_C.CONTRASTIVE.LOCAL_SHUFFLE_BN = True

# Wether to fill multiple clips (or just the first) into queue
_C.CONTRASTIVE.MOCO_MULTI_VIEW_QUEUE = False

# if sampling multiple clips per vid they need to be at least min frames apart
_C.CONTRASTIVE.DELTA_CLIPS_MIN = -math.inf

# if sampling multiple clips per vid they can be max frames apart
_C.CONTRASTIVE.DELTA_CLIPS_MAX = math.inf

# if non empty, use predictors with depth specified
_C.CONTRASTIVE.PREDICTOR_DEPTHS = []

# Wether to sequentially process multiple clips (=lower mem usage) or batch them
_C.CONTRASTIVE.SEQUENTIAL = False

# Wether to perform SimCLR loss across machines (or only locally)
_C.CONTRASTIVE.SIMCLR_DIST_ON = True

# Length of queue used in SwAV
_C.CONTRASTIVE.SWAV_QEUE_LEN = 0

# Wether to run online kNN evaluation during training
_C.CONTRASTIVE.KNN_ON = True
# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

_C.TRAIN.ONLY_VALID = False

# Dataset.
_C.TRAIN.DATASET = "imagenet"
_C.TRAIN.TRAIN_DATA_DIR = None
_C.TRAIN.TRAIN_DATA_MASK = None

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 256

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If True, use FP16 for activations
_C.TRAIN.MIXED_PRECISION = False

# Checkpoint types include `caffe2` or `pytorch`.
_C.TRAIN.CHECKPOINT_TYPE = "pytorch"


_C.TRAIN.EXTRA_MASK = None
_C.TRAIN.LOSS_WGT_SVSA = 0.0
_C.TRAIN.LOSS_WGT_CLS = 0.0
_C.TRAIN.BALANCE_WEIGHT = False

# ---------------------------------------------------------------------------- #
# Augmentation options.
# ---------------------------------------------------------------------------- #
_C.AUG = CfgNode()

# Number of repeated augmentations to used during training.
# If this is greater than 1, then the actual batch size is
# TRAIN.BATCH_SIZE * AUG.NUM_SAMPLE.
# _C.AUG.NUM_SAMPLE = 1

# # Not used if using randaug.
# _C.AUG.COLOR_JITTER = 0.4

# # RandAug parameters.
# _C.AUG.AA_TYPE = "rand-m9-n6-mstd0.5-inc1"

# # Interpolation method.
# _C.AUG.INTERPOLATION = "bicubic"

# # Probability of random erasing.
# _C.AUG.RE_PROB = 0.25

# # Random erasing mode.
# _C.AUG.RE_MODE = "pixel"

# # Random erase count.
# _C.AUG.RE_COUNT = 1

# # Do not random erase first (clean) augmentation split.
# _C.AUG.RE_SPLIT = False

# ---------------------------------------------------------------------------- #
# MipUp options.
# ---------------------------------------------------------------------------- #
_C.MIXUP = CfgNode()

# Whether to use mixup.
_C.MIXUP.ENABLE = True

# Mixup alpha.
_C.MIXUP.ALPHA = 0.8

# Cutmix alpha.
_C.MIXUP.CUTMIX_ALPHA = 1.0

# Probability of performing mixup or cutmix when either/both is enabled.
_C.MIXUP.PROB = 1.0

# Probability of switching to cutmix when both mixup and cutmix enabled.
_C.MIXUP.SWITCH_PROB = 0.5

# Label smoothing.
_C.MIXUP.LABEL_SMOOTH_VALUE = 0.1
# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = False

# Dataset for testing.
_C.TEST.DATASET = "imagenet"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 64

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# If True, convert 3D conv weights to 2D.
_C.TEST.CHECKPOINT_SQUEEZE_TEMPORAL = True

# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model name
_C.MODEL.MODEL_NAME = "MViT"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.CLS_CLASSES = 97

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"


# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.0

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"

# Activation checkpointing enabled or not to save GPU memory.
_C.MODEL.ACT_CHECKPOINT = False

_C.MODEL.DETACH_FINAL_FC = False

_C.MODEL.FREEZE_PARAM = []



_C.TEXT_NET = CfgNode()
_C.TEXT_NET.DIM = 512
_C.TEXT_NET.CONTEXT_LENGTH = 77
_C.TEXT_NET.VOCAB_SIZE = 49408
_C.TEXT_NET.WIDTH = 512
_C.TEXT_NET.HEADS = 8
_C.TEXT_NET.LAYERS = 12

_C.VISUAL_NET = CfgNode()
_C.VISUAL_NET.dim = 512
_C.VISUAL_NET.type = "ViT"
_C.VISUAL_NET.resolution = 224
_C.VISUAL_NET.patch_size = 32
_C.VISUAL_NET.width = 768
_C.VISUAL_NET.heads = 12
_C.VISUAL_NET.layers = 12
_C.VISUAL_NET.extra_token = 1

_C.VISUAL_NET.aggregation = "Transf"
_C.VISUAL_NET.atp_frames = 4
_C.VISUAL_NET.aggregation_reduce = True
_C.VISUAL_NET.aggregation_layers = 6

_C.VISUAL_NET.svsa_layers = None



# -----------------------------------------------------------------------------
# MViT options
# -----------------------------------------------------------------------------
_C.MVIT = CfgNode()

# Options include `conv`, `max`.
_C.MVIT.MODE = "conv"

# If True, perform pool before projection in attention.
_C.MVIT.POOL_FIRST = False

# If True, use cls embed in the network, otherwise don't use cls_embed in transformer.
_C.MVIT.CLS_EMBED_ON = False

# Kernel size for patchtification.
_C.MVIT.PATCH_KERNEL = [7, 7]

# Stride size for patchtification.
_C.MVIT.PATCH_STRIDE = [4, 4]

# Padding size for patchtification.
_C.MVIT.PATCH_PADDING = [3, 3]

# Base embedding dimension for the transformer.
_C.MVIT.EMBED_DIM = 96

# Base num of heads for the transformer.
_C.MVIT.NUM_HEADS = 1

# Dimension reduction ratio for the MLP layers.
_C.MVIT.MLP_RATIO = 4.0

# If use, use bias term in attention fc layers.
_C.MVIT.QKV_BIAS = True

# Drop path rate for the tranfomer.
_C.MVIT.DROPPATH_RATE = 0.1

# Depth of the transformer.
_C.MVIT.DEPTH = 16

# Dimension multiplication at layer i. If 2.0 is used, then the next block will increase
# the dimension by 2 times. Format: [depth_i: mul_dim_ratio]
_C.MVIT.DIM_MUL = []

# Head number multiplication at layer i. If 2.0 is used, then the next block will
# increase the number of heads by 2 times. Format: [depth_i: head_mul_ratio]
_C.MVIT.HEAD_MUL = []

# Stride size for the Pool KV at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_KV_STRIDE = None

# Initial stride size for KV at layer 1. The stride size will be further reduced with
# the raio of MVIT.DIM_MUL. If will overwrite MVIT.POOL_KV_STRIDE if not None.
_C.MVIT.POOL_KV_STRIDE_ADAPTIVE = None

# Stride size for the Pool Q at layer i.
# Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,]
_C.MVIT.POOL_Q_STRIDE = []

# Kernel size for Q, K, V pooling.
_C.MVIT.POOL_KVQ_KERNEL = (3, 3)

# If True, perform no decay on positional embedding and cls embedding.
_C.MVIT.ZERO_DECAY_POS_CLS = False

# If True, use absolute positional embedding.
_C.MVIT.USE_ABS_POS = False

# If True, use relative positional embedding for spatial dimentions
_C.MVIT.REL_POS_SPATIAL = True

# If True, init rel with zero
_C.MVIT.REL_POS_ZERO_INIT = False

# If True, using Residual Pooling connection
_C.MVIT.RESIDUAL_POOLING = True

# Dim mul in qkv linear layers of attention block instead of MLP
_C.MVIT.DIM_MUL_IN_ATT = True


# added by XY
_C.MVIT.REL_POS_SPATIAL_SEP = None
_C.MVIT.REL_POS_TEMPORAL = None
_C.MVIT.PATCH_2D = False
_C.MVIT.DROPOUT_RATE = 0.0
_C.MVIT.SEP_POS_EMBED = False
_C.MVIT.NORM = "layernorm"
_C.MVIT.NORM_STEM = False
_C.MVIT.CONV_Q = 1
_C.MVIT.REL_POS_V = False

_C.MVIT.WIN_SIZE = [1, 56, 56]
_C.MVIT.WIN_SIZE_STRIDE = []
_C.MVIT.SEPARATE_QKV = False

_C.MVIT.EFF_ATT_MASK_WIN = 0
_C.MVIT.EFF_ATT_MODE = 0
_C.MVIT.EFF_ATT_P = 2
_C.MVIT.EFF_MIN_WIN = 1
_C.MVIT.EFF_WIN_ADAPTIVE = False








# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()


_C.DATA.NUM_FRAMES = None
_C.DATA.FPS = None
_C.DATA.INPUT_CHANNEL_NUM = None
_C.DATA.NUM_EVAL_VIEWS = 1


# The path to the data directory.
_C.DATA.PATH_TO_DATA_DIR = ""

# If a imdb have been dumpped to a local file with the following format:
# `{"im_path": im_path, "class": cont_id}`
# then we can skip the construction of imdb and load it from the local file.
_C.DATA.PATH_TO_PRELOAD_IMDB = ""

# The mean value of pixels across the R G B channels.
_C.DATA.MEAN = [0.485, 0.456, 0.406]
# List of input frame channel dimensions.

# The std value of pixels across the R G B channels.
_C.DATA.STD = [0.229, 0.224, 0.225]


# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The relative scale range of Inception-style area based random resizing augmentation.
# If this is provided, DATA.TRAIN_JITTER_SCALES above is ignored.
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = []

# The relative aspect ratio range of Inception-style area based random resizing
# augmentation.
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = []

# If True, perform stride length uniform temporal sampling.
_C.DATA.USE_OFFSET_SAMPLING = False

# Whether to apply motion shift for augmentation.
_C.DATA.TRAIN_JITTER_MOTION_SHIFT = False

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 224

# Crop ratio for for testing. Default is 224/256.
_C.DATA.VAL_CROP_RATIO = 0.875

# If combine train/val split as training for in21k
_C.DATA.IN22K_TRAINVAL = False

# If not None, use IN1k as val split when training in21k
_C.DATA.IN22k_VAL_IN1K = ""
# if True, sample uniformly in [1 / max_scale, 1 / min_scale] and take a
# reciprocal to get the scale. If False, take a uniform sample from
# [min_scale, max_scale].
_C.DATA.INV_UNIFORM_SAMPLE = False



# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.00025

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 1e-6

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 0.05

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 70.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 1e-8

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# If True, start from the peak cosine learning rate after warm up.
_C.SOLVER.COSINE_AFTER_WARMUP = True

# If True, perform no weight decay on parameter with one dimension (bias term, etc).
_C.SOLVER.ZERO_WD_1D_PARAM = True

# Clip gradient at this value before optimizer update
_C.SOLVER.CLIP_GRAD_VAL = None

# Clip gradient at this norm before optimizer update
_C.SOLVER.CLIP_GRAD_L2NORM = None

# The layer-wise decay of learning rate. Set to 1. to disable.
_C.SOLVER.LAYER_DECAY = 1.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.TASK = "verb"
_C.TEMPERATURE = 0.07
_C.LOSS_TYPE = "local"

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 8

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./logs"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 0

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True




# ---------------------------------------------------------------------------- #
# New for slowfast
# ---------------------------------------------------------------------------- #



_C.BN = CfgNode()

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized. `NUM_SYNC_DEVICES` cannot be larger than number of
# devices per machine; if global sync is desired, set `GLOBAL_SYNC`.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.NUM_SYNC_DEVICES = 1

# Parameter for NaiveSyncBatchNorm. Setting `GLOBAL_SYNC` to True synchronizes
# stats across all devices, across all machines; in this case, `NUM_SYNC_DEVICES`
# must be set to None.
# By default ONLY applies to NaiveSyncBatchNorm3d; consider also setting
# CONTRASTIVE.BN_SYNC_MLP if appropriate.
_C.BN.GLOBAL_SYNC = False




_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

#  If true, initialize the final conv layer of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_CONV = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1], [1], [1], [1]]




_C.X3D = CfgNode()

# Width expansion factor.
_C.X3D.WIDTH_FACTOR = 1.0

# Depth expansion factor.
_C.X3D.DEPTH_FACTOR = 1.0

# Bottleneck expansion factor for the 3x3x3 conv.
_C.X3D.BOTTLENECK_FACTOR = 1.0  #

# Dimensions of the last linear layer before classificaiton.
_C.X3D.DIM_C5 = 2048

# Dimensions of the first 3x3 conv layer.
_C.X3D.DIM_C1 = 12

# Whether to scale the width of Res2, default is false.
_C.X3D.SCALE_RES2 = False

# Whether to use a BatchNorm (BN) layer before the classifier, default is false.
_C.X3D.BN_LIN5 = False

# Whether to use channelwise (=depthwise) convolution in the center (3x3x3)
# convolution operation of the residual blocks.
_C.X3D.CHANNELWISE_3x3x3 = True




_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# Instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"


# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]



_C.MODEL.ARCH = "slowfast"
_C.MODEL.SINGLE_PATHWAY_ARCH = [
    "2d",
    "c2d",
    "i3d",
    "slow",
    "x3d",
    "mvit",
    "maskmvit",
]



_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5

























def assert_and_infer_cfg(cfg):
    # TRAIN assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.NUM_GPUS == 0 or cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.WARMUP_START_LR *= cfg.NUM_SHARDS
        cfg.SOLVER.COSINE_END_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _C.clone()
