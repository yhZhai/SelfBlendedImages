import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
import warnings
from torchvision.transforms.functional import normalize, resize, to_pil_image
from matplotlib import cm


warnings.filterwarnings("ignore")

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def main(args):
    device = torch.device("cuda")

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()

    face_list, idx_list = extract_frames(args.input_video, args.n_frames, face_detector)

    with torch.no_grad():
        img = torch.tensor(face_list).to(device).float() / 255
        pred = model(img).softmax(1)[:, 1]

        # get cam
        if args.get_cam:
            cam = model.get_cam(img).softmax(-1)[:,:,:,1]
            for i in range(cam.shape[0]):
                result = overlay_mask(to_pil_image(img[i]), to_pil_image(cam[i], mode="F"), alpha=0.5)
                plt.imshow(result)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"tmp/cam_{i}.png", dpi=300)
                plt.close()

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

    print(f"fakeness: {pred:.4f}")


if __name__ == "__main__":

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_name", type=str)
    parser.add_argument("-i", dest="input_video", type=str)
    parser.add_argument("-n", dest="n_frames", default=32, type=int)
    parser.add_argument("--get_cam", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
