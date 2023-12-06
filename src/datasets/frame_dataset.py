import sys
import numpy as np
from pathlib import Path
from PIL import Image
import albumentations as alb
from torchvision import transforms as T
from loguru import logger
from torch.utils.data import ConcatDataset, Dataset

sys.path.append(str(Path(__file__).parent.parent))
try:
    from .utils.funcs import IoUfrom2bboxes, crop_face, RandomDownScale
except:
    from utils.funcs import IoUfrom2bboxes, crop_face, RandomDownScale


class FrameDataset(Dataset):
    def __init__(self, real_path_list, fake_path_list, image_size=256, multiclass=False):
        self.real_path_list = real_path_list
        self.fake_path_list = fake_path_list

        self.real_frame_path_list = []
        self.fake_frame_path_list = []
        self.label_list = []

        for real_path in real_path_list:
            frames = list(Path(real_path).rglob("*.png"))
            self.real_frame_path_list.extend(frames)
            self.label_list.extend([0] * len(frames))

        for i, fake_path in enumerate(fake_path_list):
            frames = list(Path(fake_path).rglob("*.png"))
            self.fake_frame_path_list.extend(frames)
            if multiclass:
                self.label_list.extend([i + 1] * len(frames))
            else:
                self.label_list.extend([1] * len(frames))


        logger.info(
            "num real frames: {}, num fake frames: {}".format(
                len(self.real_frame_path_list), len(self.fake_frame_path_list)
            )
        )

        self.frame_path_list = self.real_frame_path_list + self.fake_frame_path_list

        self.to_tensor_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomResizedCrop(
                    image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.8, 1.2),
                    interpolation=Image.BICUBIC,
                ),
                T.ToTensor(),
            ]
        )
        self.transform = alb.Compose(
            [
                alb.Compose(
                    [
                        alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                        alb.HueSaturationValue(
                            hue_shift_limit=(-0.3, 0.3),
                            sat_shift_limit=(-0.3, 0.3),
                            val_shift_limit=(-0.3, 0.3),
                            p=1,
                        ),
                        alb.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.1),
                            contrast_limit=(-0.1, 0.1),
                            p=1,
                        ),
                    ],
                    p=1,
                ),
                alb.OneOf(
                    [
                        RandomDownScale(p=1),
                        alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                    ],
                    p=1,
                ),
            ],
            p=1.0,
        )

    def __len__(self):
        return len(self.frame_path_list)

    def __getitem__(self, index):
        frame_path = self.frame_path_list[index]
        label = self.label_list[index]
        frame = Image.open(frame_path).convert("RGB")

        frame = self.transform(image=np.array(frame))["image"]

        frame = Image.fromarray(frame)
        frame = self.to_tensor_transform(frame)

        return {"img": frame, "label": label}


if __name__ == "__main__":
    real_path_list = ["data/eval4/PRISTINE"]
    fake_path_list = ["data/eval4/DAGAN"]
    dataset = FrameDataset(real_path_list, fake_path_list)
    print(len(dataset))
    print(dataset[0])
