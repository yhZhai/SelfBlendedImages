from pathlib import Path
from typing import List
import random

import cv2
from PIL import Image
import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T



class ImagePairDataset(Dataset):
    def __init__(
        self,
        split: str,
        real_paths: List[str],
        fake_paths: List[str],
        img_suffix: str = ".png",
    ):
        self.split = split
        self.real_paths = real_paths
        self.fake_paths = fake_paths
        self.img_suffix = img_suffix

        self.real_video = {}
        for real_path in real_paths:
            real_path = Path(real_path)
            for subfolder in real_path.iterdir():
                if not subfolder.is_dir():
                    continue
                frames = sorted(list(subfolder.glob(f"*{img_suffix}")))
                self.real_video[subfolder.as_posix()] = frames

        self.fake_video = {}
        for fake_path in fake_paths:
            fake_path = Path(fake_path)
            for subfolder in fake_path.iterdir():
                if not subfolder.is_dir():
                    continue
                frames = sorted(list(subfolder.glob(f"*{img_suffix}")))
                self.fake_video[subfolder.as_posix()] = frames

        self.num_real_video = len(self.real_video)
        self.num_fake_video = len(self.fake_video)
        self.num_videos = self.num_real_video + self.num_fake_video
        self.labels = [0] * self.num_real_video + [1] * self.num_fake_video

        print(f"Found {self.num_real_video} real videos and {self.num_fake_video} fake videos")

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(296),
                A.CenterCrop(296, 296),  # TODO change according to face location
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            ],
            additional_targets={"image1": "image"},
        )

    def __len__(self):
        return self.num_videos

    def __getitem__(self, index):
        label = self.labels[index]
        if index >= self.num_real_video:
            key = list(self.fake_video.keys())[index - self.num_real_video]
            video = self.fake_video[key]
        else:
            key = list(self.real_video.keys())[index]
            video = self.real_video[key]

        num_frames = len(video)
        first_frame_idx = random.randint(0, num_frames - 1)
        first_frame = video[first_frame_idx]
        second_frame = video[first_frame_idx + 1]

        first_frame = cv2.imread(first_frame.as_posix())
        second_frame = cv2.imread(second_frame.as_posix())
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=first_frame, image1=second_frame)
        first_frame = transformed["image"]
        second_frame = transformed["image1"]

        first_frame = torch.from_numpy(first_frame.transpose(2, 0, 1))
        second_frame = torch.from_numpy(second_frame.transpose(2, 0, 1))
        
        return {"first_frame": first_frame, "second_frame": second_frame, "label": label}


class FlowValDataset(Dataset):
    def __init__(self, dataset_name: str, num_frames: int = 5, flow_suffix: str = ".flo"):
        dataset_name = dataset_name.lower()
        assert dataset_name in ["ffiw", "cdf"], f"Invalid dataset name {dataset_name}"

        self.dataset_name = dataset_name
        self.num_frames = num_frames
        self.flow_suffix = flow_suffix

        if dataset_name == "ffiw":
            flow_list, label_list = self._init_ffiw()
        elif dataset_name == "cdf":
            flow_list, label_list = self._init_cdf()
        
        self.flow_list = flow_list
        self.label_list = label_list
        
        self.A_transform = A.Compose(
            [
                A.SmallestMaxSize(256),
                A.CenterCrop(256, 256),  # TODO change according to face location
            ]
        )
        # self.T_transform = T.FiveCrop(256)
    
    def _init_ffiw(self):
        pass

    def _init_cdf(self, video_list_txt="data/Celeb-DF-v2/List_of_testing_videos.txt"):
        
        folder_list = []
        label_list = []
        with open(video_list_txt) as f:

            for data in f:
                # print(data)
                line = data.split()
                # print(line)
                path = line[1].split("/")
                folder_list += ["data/Celeb-DF-v2/" + path[0] + "/flow/" + Path(path[1]).stem]
                label_list += [1 - int(line[0])]
        return folder_list, label_list
    
    def __len__(self):
        return len(self.flow_list)
    
    def __getitem__(self, index):
        flow_folder_path = self.flow_list[index]
        label = self.label_list[index]

        flow_files = list(Path(flow_folder_path).rglob(f"*{self.flow_suffix}"))
        flow_files = sorted(flow_files, key=lambda x: int(Path(x).stem))
        sel_flow_files = []
        if len(flow_files) <= self.num_frames:
            sel_flow_files = flow_files
        else:
            sel_indices = np.linspace(0, len(flow_files) - 1, self.num_frames, dtype=int)
            sel_flow_files = [flow_files[i] for i in sel_indices]
        
        flow_list = []
        for sel_flow_file in sel_flow_files:
            flow = readFlow(sel_flow_file)
            flow = self.A_transform(image=flow)["image"]
            # flow = self.T_transform(torch.from_numpy(flow.transpose(2, 0, 1)))
            # flow = torch.stack(flow)

            flow = torch.from_numpy(flow).permute(2, 0, 1)
            flow_list.append(flow)

        return {"flow": flow_list, "label": label}


if __name__ == "__main__":
    dataset = ImagePairDataset(
        "train",
        ["data/FaceForensics++/original_sequences/youtube/c23/frames"],
        ["data/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames"],
    )
    # dataset = FlowValDataset("cdf")

    for item in dataset:
        print("a")
