from pathlib import Path
from typing import List

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


class FlowDataset(Dataset):
    def __init__(
        self,
        split: str,
        real_paths: List[str],
        fake_paths: List[str],
        flow_suffix: str = ".flo",
    ):
        self.split = split
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.flow_suffix = flow_suffix

        num_real_videos = 0
        real_flow_files = []
        for real_path in real_paths:
            real_path = Path(real_path)
            subfolder_count = sum(1 for item in real_path.iterdir() if item.is_dir())
            num_real_videos += subfolder_count

            flow_file = real_path.rglob(f"*{flow_suffix}")
            real_flow_files.extend(list(flow_file))

        num_fake_videos = 0
        fake_flow_files = []
        for fake_path in fake_paths:
            fake_path = Path(fake_path)
            subfolder_count = sum(1 for item in fake_path.iterdir() if item.is_dir())
            num_fake_videos += subfolder_count

            flow_file = fake_path.rglob(f"*{flow_suffix}")
            fake_flow_files.extend(list(flow_file))

        self.flow_files = real_flow_files + fake_flow_files
        self.labels = [0] * len(real_flow_files) + [1] * len(fake_flow_files)

        print(f"Found {num_real_videos} real videos and {num_fake_videos} fake videos")
        print(
            f"Found {len(real_flow_files)} real flow files and {len(fake_flow_files)} fake flow files"
        )

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(296),
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(256, 256),
            ]
        )

    def __len__(self):
        return len(self.real_paths) + len(self.fake_paths)

    def __getitem__(self, index):
        file = self.flow_files[index]
        label = self.labels[index]
        flow = readFlow(file)
        flow = self.transform(image=flow)["image"]
        flow = torch.from_numpy(flow).permute(2, 0, 1)
        return {"flow": flow, "label": label}


if __name__ == "__main__":
    dataset = FlowDataset(
        "train",
        ["data/FaceForensics++/original_sequences/youtube/c23/flow"],
        ["data/FaceForensics++/manipulated_sequences/Deepfakes/c23/flow"],
    )

    for item in dataset:
        print("a")
