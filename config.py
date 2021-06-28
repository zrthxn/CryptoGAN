from datetime import datetime
from typing import List

defaults = {
    "avail_models": ["anc", "cryptonet", "cryptonet_anc"],
    "model": "anc",
    "debug": False,
    "tensorboard": False,

    # Default config options for Trainer
    "dropout": 0.1,
    "training": {
        "run": str(datetime.now()),
        "batches": 12800, 
        "epochs": 10
    },

    # Default config options for ANC
    "anc": {
        "blocksize": 16,
        "batchlen": 64,
        "alice_lr": 0.001,
        "eve_lr": 0.001,
    },
    
    # Default config options for Cryptonet
    "cryptonet": {
        "blocksize": 16,
        "batchlen": 64,
        "alice_lr": 0.001,
        "eve_lr": 0.001,
    },

    # Default config options for Cryptonet+ANC
    "cryptonet_anc": {
        "blocksize": 16,
        "batchlen": 64,
        "alice_lr": 0.001,
        "eve_lr": 0.001,
    },

    "save_model": True,
}

def build_config(argv: List[str]):
    global defaults
    config = dict()

    for arg in argv:
        if arg.find("=") == -1:
            continue
        name, value = arg.split("=")
        keys = name.split("-")
        
        config = tree_traverse(defaults, keys, value)

    defaults = config


def tree_traverse(tree: dict, keys: List[str], value):
    key = keys.pop(0)

    if key in tree.keys():
        tree[key] = tree_traverse(tree[key], keys, value) if type(tree[key]) == dict else type(tree[key])(value)
        return tree
