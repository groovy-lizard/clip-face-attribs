"""Utilitary functions"""
import json
import sys
from pathlib import Path
import torch
import open_clip
import numpy as np


def prep_folders(path):
    """Make folders given path, if they does not exists

    :param path: directory path
    :type path: str
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_open_clip(backbone: str, datasource: str) -> dict:
    """Initialize a pretrained openCLIP model

    :param backbone: Name of the model's backbone
    :type backbone: str
    :param datasource: Name of the data source where model was trained
    :type datasource: str
    :return: model dictionary with model, preprocess, device and tokenizer
    :rtype: dict[obj]
    """
    available_models = open_clip.list_pretrained()

    if (backbone, datasource) in available_models:
        print(f"Loading model: ({backbone}, {datasource})")
    else:
        print(
            f"ERROR: ({backbone}, {datasource}) not found in ")
        print("open_clip.list_pretrained().")
        sys.exit(-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocessing = open_clip.create_model_and_transforms(
        backbone, pretrained=datasource, device=device)
    print(f"Done! ({backbone}, {datasource}) loaded to {device} device")
    model_dict = {}
    model_dict = {"Model": model,
                  "Preprocessing": preprocessing,
                  "Device": device,
                  "Tokenizer": open_clip.tokenizer.tokenize}
    return model_dict


def measure_race_gap(mlist: list) -> float:
    """Measure the Gap of race accuracies

    :param mlist: list of accuracies by race
    :type mlist: list
    :return: Gap measurement
    :rtype: float
    """
    metrics = np.array(mlist)
    min_val = metrics.min()
    # metrics = np.delete(metrics, metrics.argmin())
    avg = np.average(metrics)
    return np.round(avg - min_val, 4)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data
