import random
import argparse
import warnings
from pathlib import Path

import h5py
from torch.utils.data.dataset import ConcatDataset, Dataset
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, roc_auc_score

from datasets import *
from model import Detector
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames

warnings.filterwarnings("ignore")

class HDF5Dataset(Dataset):
    def __init__(self, video_list, dataset_name, num_frames: int, cache_root: str = ".cache"):
        super().__init__()
        self.video_list = video_list
        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.cache_root = cache_root
    
    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, index):
        filename = self.video_list[index]
        cache_path = Path(self.cache_root, self.dataset_name.lower(), Path(filename).stem + ".hdf5")
        assert cache_path.exists()
        with h5py.File(cache_path, "r") as f:
            face_list = f["frames"]
            idx_list = f["indices"]
            sel_indices = random.sample(range(face_list.shape[0]), self.num_frames)
            sel_indices = sorted(sel_indices)
            face_list = np.array([face_list[i] for i in sel_indices])
            idx_list = np.array([idx_list[i] for i in sel_indices])
        
        return face_list, idx_list


def get_frames(filename, num_frames: int, face_detector, dataset_name: str, cache_root: str = ".cache"):
    cache_path = Path(cache_root, dataset_name.lower(), Path(filename).stem + ".hdf5")
    if cache_path.exists():
        try:
            with h5py.File(cache_path, "r") as f:
                face_list = f["frames"]
                idx_list = f["indices"]
                sel_indices = random.sample(range(face_list.shape[0]), num_frames)
                sel_indices = sorted(sel_indices)
                face_list = np.array([face_list[i] for i in sel_indices])
                idx_list = np.array([idx_list[i] for i in sel_indices])
        except Exception as e:
            print("Error in reading cache file:", e)
            face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)
        return face_list, idx_list
    else:
        face_list, idx_list = extract_frames(filename, args.n_frames, face_detector)
    return face_list, idx_list


def main(args):
    device = torch.device("cuda")

    model = Detector()
    model = model.to(device)
    cnn_sd = torch.load(args.weight_name, map_location="cpu")["model"]
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
 
    use_hdf5 = False
    if Path(args.cache_root, args.dataset.lower()).exists():
        use_hdf5 = True
        dataset = HDF5Dataset(video_list, args.dataset.lower(), args.n_frames, args.cache_root)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
        print("Using HDF5 dataset")
    else:
        dataloader = video_list
        print("Using raw dataset")

    output_list = []
    for data in tqdm(dataloader):
        try:
            if use_hdf5:
                face_list, idx_list = data
                face_list = face_list.squeeze(0)
                idx_list = idx_list.squeeze(0)
            else:
                filename = data
                face_list, idx_list = get_frames(filename, args.n_frames, face_detector, args.dataset)

            with torch.no_grad():
                img = torch.tensor(face_list).to(device).float() / 255
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

    seed = 1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", dest="weight_name", type=str)
    parser.add_argument("-d", dest="dataset", type=str)
    parser.add_argument("-n", dest="n_frames", default=32, type=int)
    parser.add_argument("--cache_root", default=".cache", type=str)
    args = parser.parse_args()

    main(args)
