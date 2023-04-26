# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from .build import build_dataset, DATASET_REGISTRY  # noqa

from .CGQA import CGQA
from .VG import VG
# from .Ego4d_LongAction import Ego4dAction

from .ExtrasampleDataset import ExtrasampleDataset
from .ReducedSampleDataset import ReducedSampleDataset
from .One4AllDataset_attr import One4AllDataset_Attr