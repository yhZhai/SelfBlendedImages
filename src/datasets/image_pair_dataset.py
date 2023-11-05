import random
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image
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

        print(
            f"Found {self.num_real_video} real videos and {self.num_fake_video} fake videos"
        )

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(296),
                A.CenterCrop(296, 296),  # TODO change according to face location
                A.HorizontalFlip(p=0.5),
                A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            ],
            additional_targets={"image1": "image"},
        )
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
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
        first_frame_idx = random.randint(0, num_frames - 2)
        first_frame = video[first_frame_idx]
        second_frame = video[first_frame_idx + 1]

        first_frame = cv2.imread(first_frame.as_posix())
        second_frame = cv2.imread(second_frame.as_posix())
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=first_frame, image1=second_frame)
        first_frame = transformed["image"]
        second_frame = transformed["image1"]
        first_frame = Image.fromarray(first_frame)
        second_frame = Image.fromarray(second_frame)

        first_frame = self.normalize(first_frame)
        second_frame = self.normalize(second_frame)

        return {
            "first_frame": first_frame,
            "second_frame": second_frame,
            "label": label,
        }


class ValImagePairDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        img_suffix: str = ".png",
    ):
        dataset_name = dataset_name.lower()
        assert dataset_name in ["ffiw", "cdf"], f"Invalid dataset name {dataset_name}"

        self.dataset_name = dataset_name
        self.img_suffix = img_suffix

        if dataset_name == "ffiw":
            frame_list, label_list = self._init_ffiw()
        elif dataset_name == "cdf":
            frame_list, label_list = self._init_cdf()

        self.frame_list = frame_list
        self.label_list = label_list

        self.transform = A.Compose(
            [
                A.SmallestMaxSize(256),
                A.CenterCrop(256, 256),  # TODO change according to face location
            ]
        )
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _init_ffiw(self):
        raise NotImplementedError

    def _init_cdf(self, video_list_txt="data/Celeb-DF-v2/List_of_testing_videos.txt"):

        folder_list = []
        label_list = []
        with open(video_list_txt) as f:

            for data in f:
                # print(data)
                line = data.split()
                # print(line)
                path = line[1].split("/")
                folder_list += [
                    "data/Celeb-DF-v2/" + path[0] + "/frames/" + Path(path[1]).stem
                ]
                label_list += [1 - int(line[0])]
        return folder_list, label_list

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, index):
        frame_folder_path = self.frame_list[index]
        label = self.label_list[index]

        frame_files = list(Path(frame_folder_path).rglob(f"*{self.img_suffix}"))
        frame_files = sorted(frame_files, key=lambda x: int(Path(x).stem))
        num_frames = len(frame_files)
        first_frame_idx = random.randint(0, num_frames - 2)
        first_frame = frame_files[first_frame_idx]
        second_frame = frame_files[first_frame_idx + 1]

        first_frame = cv2.imread(first_frame.as_posix())
        second_frame = cv2.imread(second_frame.as_posix())
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=first_frame, image1=second_frame)
        first_frame = transformed["image"]
        second_frame = transformed["image1"]
        first_frame = Image.fromarray(first_frame)
        second_frame = Image.fromarray(second_frame)

        first_frame = self.normalize(first_frame)
        second_frame = self.normalize(second_frame)

        return {
            "first_frame": first_frame,
            "second_frame": second_frame,
            "label": label,
        }


if __name__ == "__main__":
    # dataset = ImagePairDataset(
    #     "train",
    #     ["data/FaceForensics++/original_sequences/youtube/c23/frames"],
    #     ["data/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames"],
    # )
    dataset = ValImagePairDataset("cdf")

    for item in dataset:
        print("a")
