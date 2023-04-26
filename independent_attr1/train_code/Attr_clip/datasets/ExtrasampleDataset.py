import json
import random
import torch.utils.data as tdata

from datasets.build import DATASET_REGISTRY, build_dataset


EXTRA_DIR = {
    "Epic":             "data/epic-kitchens",
    "FiftySalads":      "data/50salad",
    "Breakfast":        "data/breakfast",
    "IKEA":             "data/ikea_asm_dataset_public",
    "EgteaGaze":        "data/EGTEA_GAZE+",
    "CharadesEgo":      "data/charades-ego",
    "Ego4dAction":      "data/Ego4d/ego4d-fho/",
    "Ego4dActionNew":   "data/Ego4d/ego4d-fho/",
    "SthElse":          "data/something-else/",
}




@DATASET_REGISTRY.register()
class ExtrasampleDataset(tdata.Dataset):
    def __init__(self, cfg, mode):
        """ConcatDataset with interface `countbbox`"""
        self.datasets = {}

        if mode == "train":
            ds_name, path = cfg.TRAIN.TRAIN_DATA_DIR.split(":")

            # mainbody
            cfg.DATA.PATH_TO_DATA_DIR = path
            self.datasets[ds_name] = build_dataset(ds_name, cfg, mode)
            self.idx_list = [(ds_name, i) for i in range(len(self.datasets[ds_name]))]
            num_main = len(self.idx_list)

            # extra
            with open(cfg.TRAIN.EXTRA_MASK, "r") as fp:
                extra_idx = json.load(fp)

            dataset_list = {k for k, _ in extra_idx}
            for name in dataset_list:
                cfg.DATA.PATH_TO_DATA_DIR = EXTRA_DIR[name]
                dset = build_dataset(name, cfg, mode)
                self.datasets[name] = dset

            self.idx_list += extra_idx
            print(f"{ds_name} (num_main) + extra ({'/'.join(dataset_list)}, {len(extra_idx)}), total size: {len(self.idx_list)}")

        else :
            ds_name, path = cfg.TRAIN.TRAIN_DATA_DIR.split(":")
            print(f"val, only {ds_name}")
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
