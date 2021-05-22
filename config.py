from typing import List

defaults = {
    "avail_models": ["anc", "cryptonet"],
    "model": "anc",
    "debug": False,
    "tensorboard": False,

    # Default config options for Trainer
    "dropout": 0.1,
    "training": {
        "batches": 12800, 
        "epochs": 10
    },

    "anc": {
        "blocksize": 16,
        "batchlen": 64,
        "alice_lr": 0.001,
        "eve_lr": 0.001,
    },

    "cryptonet": {
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
        name, value = arg.split("=")
        keys = name.split("-")
        
        config = tree_traverse(defaults, keys, value)

    defaults = config


def tree_traverse(tree: dict, keys: List[str], value):
    key = keys.pop(0)

    if key in tree.keys():
        tree[key] = tree_traverse(tree[key], keys, value) if type(tree[key]) == dict else type(tree[key])(value)
        return tree
