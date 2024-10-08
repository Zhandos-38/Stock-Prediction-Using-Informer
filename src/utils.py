import numpy as np
import pandas as pd
import yaml
from importlib import import_module

def load_yaml(file_name: str) -> dict:
    with open(file_name, "r") as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    return config

def build_object(type: str, imported_module: str) -> object:
    module = import_module(imported_module)
    return getattr(module, type)

def destandardize(value, mean, std):
    return value * std + mean

def directional_accuracy(preds, labels, last):
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    last = last.cpu().detach().numpy()
    preds = np.concatenate([last, preds], axis=1)
    labels = np.concatenate([last, labels], axis=1)
    preds = np.diff(preds, axis=1)
    labels = np.diff(labels, axis=1)
    preds = np.sign(preds)
    labels = np.sign(labels)
    signs = np.sum(preds == labels)
    total = np.prod(preds.shape)
    return signs, total