import json
import random
import torch.utils.data as tdata

from datasets.build import DATASET_REGISTRY, build_dataset


EXTRA_DIR = {
    "Epic": "/ssd/FAST_DATA/epic-kitchens",
    "FiftySalads": "/hdd/DATA/50salad",
    "Breakfast": "/hdd/DATA/breakfast",
    "IKEA": "/hdd/DATA/ikea_asm_dataset_public",
    "EgteaGaze": "/ssd/FAST_DATA/EGTEA_GAZE+",
    "CharadesEgo": "/hdd/DATA/charades-ego",
    "Ego4dAction": "/hdd/DATA/Ego4d/ego4d-fho/",
    "Ego4dActionNew": "/hdd/DATA/Ego4d/ego4d-fho/",
    "SthElse": "/hdd/DATA/something-else/",
}


@DATASET_REGISTRY.register()
class ReducedSampleDataset(tdata.Dataset):
    def __init__(self, cfg, mode):
        """ConcatDataset with interface `countbbox`"""
        self.datasets = {}

        if mode == "train":
            ds_name, path = cfg.TRAIN.TRAIN_DATA_DIR.split(":")

            # # mainbody
            # cfg.DATA.PATH_TO_DATA_DIR = path
            # self.datasets[ds_name] = build_dataset(ds_name, cfg, mode)
            # self.idx_list = [(ds_name, i) for i in range(len(self.datasets[ds_name]))]
            # num_main = len(self.idx_list)

            with open(cfg.TRAIN.EXTRA_MASK, "r") as fp:
                extra_idx = json.load(fp)

            dataset_list = {k for k, _ in extra_idx}
            for name in dataset_list:
                cfg.DATA.PATH_TO_DATA_DIR = EXTRA_DIR[name]
                dset = build_dataset(name, cfg, mode)
                self.datasets[name] = dset

            self.idx_list = extra_idx
            print(f"({'/'.join(dataset_list)}, {len(extra_idx)})")

        else :
            ds_name, path = cfg.TRAIN.TRAIN_DATA_DIR.split(":")
            print(f"val {ds_name}")
            cfg.DATA.PATH_TO_DATA_DIR = path
            self.datasets[ds_name] = build_dataset(ds_name, cfg, mode)
            self.idx_list = [(k, i) for k, ds in self.datasets.items() for i in range(len(ds))]
            print(f"concat dataset from {list(self.datasets.keys())}, total size: {len(self.idx_list)}")

            self.prompt_token_per_class = self.datasets[ds_name].prompt_token_per_class

        self.ds_name = ds_name
        self.mode = mode

    
    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        name, idx = self.idx_list[index]
        batch = self.datasets[name][idx]
        if self.mode == "train" and name != self.ds_name:
            batch[-2] = -1  # is extra sample
        return batch