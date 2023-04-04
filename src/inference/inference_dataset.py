import sys
import warnings
from datetime import datetime
import argparse
import random
import shutil
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score
from albumentations.pytorch.functional import img_to_tensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

current_path = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
sys.path.append("/home/csgrad/yzhai6/Projects/SelfBlendedImages/retinaface")

from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from networks.xception import TransferModel

warnings.filterwarnings("ignore")


def main(args):

    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    if args.dataset == "FFIW":
        video_list, target_list = init_ffiw()
    elif args.dataset == "FF":
        video_list, target_list = init_ff()
    elif args.dataset == "DFD":
        video_list, target_list = init_dfd()
    elif args.dataset == "DFDC":
        video_list, target_list = init_dfdc()
    elif args.dataset == "DFDCP":
        video_list, target_list = init_dfdcp()
    elif args.dataset == "CDF":
        video_list, target_list = init_cdf()
    else:
        NotImplementedError

    output_list = []
    for filename in tqdm(video_list):
        try:
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float() / 255
                mean = torch.tensor(IMAGENET_DEFAULT_MEAN).to(device).view(1, 3, 1, 1)
                std = torch.tensor(IMAGENET_DEFAULT_STD).to(device).view(1, 3, 1, 1)
                img = (img - mean) / std

                pred = model(img).softmax(1)[:, 1]

            pred_list = []
            idx_img = -1
            for i in range(len(pred)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res = np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i] = max(pred_list[i])
            pred = pred_res.mean()
        except Exception as e:
            print(e)
            pred = 0.5
        output_list.append(pred)

    auc = roc_auc_score(target_list, output_list)
    print(f"{args.dataset}| AUC: {auc:.4f}")


if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda")

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_name", type=str)
    parser.add_argument("-d", dest="dataset", type=str)
    parser.add_argument("-n", dest="n_frames", default=32, type=int)
    args = parser.parse_args()

    main(args)
