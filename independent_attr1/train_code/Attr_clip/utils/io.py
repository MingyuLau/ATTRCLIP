import pickle
import json
import os
from typing import Any, List

__all__ = [
    "load_json",
    "dump_json",
    "load_pickle_bin",
    "dump_pickle_bin",
    "load_txt",
    "dump_txt",
]


def load_json(*paths: str) -> Any:
    """Read json file give filepath. The input paths will be joined"""
    with open(os.path.join(*paths), "r") as fp:
        return json.load(fp)

def dump_json(obj: Any, *paths: str):
    """Write json file give filepath. The input paths will be joined"""
    with open(os.path.join(*paths), "w") as fp:
        json.dump(obj, fp)

def load_pickle_bin(*paths: str) -> Any:
    """Read pickle file give filepath. The input paths will be joined"""
    with open(os.path.join(*paths), "rb") as fp:
        return pickle.load(fp)

def dump_pickle_bin(obj: Any, *paths: str):
    """Write pickle file give filepath. The input paths will be joined"""
    with open(os.path.join(*paths), "wb") as fp:
        pickle.dump(obj, fp)

def load_txt(*paths: str) -> Any:
    """Read non-empty lines in txt file give filepath. The input paths will be joined"""
    res = []
    with open(os.path.join(*paths), "r") as fp:
        for line in fp:
            line = line.strip()
            if len(line)>0:
                res.append(line)
    return res

def dump_txt(str_list: List[str], *paths: str):
    """Write lines to txt file give filepath. The input paths will be joined"""
    with open(os.path.join(*paths), "w") as fp:
        for line in str_list:
            line = line.strip()
            fp.write(line+"\n")