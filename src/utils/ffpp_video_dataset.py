from typing import List
from pathlib import Path
from glob import glob
import json
import random

from rich import print
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.functional import img_to_tensor
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from funcs import IoUfrom2bboxes, crop_face, custom_crop_face
except:
    from utils.funcs import IoUfrom2bboxes, crop_face, custom_crop_face


class FFPPVideoDataset(Dataset):
    def __init__(
        self,
        phase: str,
        image_size: int = 224,
        n_frames: int = 8,
        fake_subsets: List = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
        compression: str = "c23",
        dataset_root: str = "data/FaceForensics++/",
        landmark_path: str = "/landmarks/",
        patch_size: int = 32,
        verbose: bool = False,
    ):

        assert phase in ["train", "val", "test"]
        for item in fake_subsets:
            assert item in [
                "Deepfakes",
                "Face2Face",
                "FaceShifter",
                "FaceSwap",
                "NeuralTextures",
            ]
        assert compression in ["c23", "raw"]

        self.phase = phase
        self.image_size = image_size
        self.n_frames = n_frames
        self.fake_subsets = fake_subsets
        self.compression = compression
        self.dataset_root = Path(dataset_root)
        self.landmark_path = landmark_path
        self.patch_size = patch_size
        self.verbose = verbose

        self.image_dict, self.label_dict = self._get_datalist()
        self.transform = self.get_transforms()

    def get_transforms(self):
        return A.ReplayCompose(
            [
                A.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=(-0.3, 0.3),
                    sat_shift_limit=(-0.3, 0.3),
                    val_shift_limit=(-0.3, 0.3),
                    p=0.3,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3
                ),
                A.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
                # TODO should add gaussian blur?
                # TODO should add the following?
                # alb.OneOf(
                #     [
                #         RandomDownScale(p=1),
                #         alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
                #     ],
                #     p=1,
                # ),
            ]
        )

    def _get_datalist(self):
        image_dict, label_dict = {}, {}

        file_list = []
        with open(self.dataset_root / f"{self.phase}.json", "r") as f:
            list_dict = json.load(f)
        for i in list_dict:
            file_list += i

        # for authentic videos
        folder_list = sorted(
            glob(
                str(
                    self.dataset_root
                    / "original_sequences"
                    / "youtube"
                    / self.compression
                    / "frames/*"
                )
            )
        )
        for i in range(len(folder_list)):
            if Path(folder_list[i]).stem[:3] not in file_list:
                continue
            images_temp = sorted(glob(folder_list[i] + "/*.png"))
            image_dict[Path(folder_list[i]).stem] = images_temp
            label_dict[Path(folder_list[i]).stem] = 0

        print(f"Found {len(image_dict)} authentic videos.")

        # for fake videos
        cnt = 0
        for fake_subset in self.fake_subsets:
            folder_list = sorted(
                glob(
                    str(
                        self.dataset_root
                        / "manipulated_sequences"
                        / fake_subset
                        / self.compression
                        / "frames/*"
                    )
                )
            )
            for i in range(len(folder_list)):
                if Path(folder_list[i]).stem[:3] not in file_list:
                    continue
                images_temp = sorted(glob(folder_list[i] + "/*.png"))
                image_dict[Path(folder_list[i]).stem] = images_temp
                label_dict[Path(folder_list[i]).stem] = 1
                cnt += 1

        print(f"Found {cnt} fake videos.")
        return image_dict, label_dict

    def __len__(self):
        return len(self.image_dict)

    def __getitem__(self, index):
        flag = True
        while flag:
            try:
                video_name = list(self.image_dict.keys())[index]
                total_num_frames = len(self.image_dict[video_name])
                max_start_frame = total_num_frames - self.n_frames
                start_frame = random.randint(0, max_start_frame)
                frame_path_list = self.image_dict[video_name][
                    start_frame : start_frame + self.n_frames
                ]

                # load
                image_list = []
                landmark_list = []
                patch_list = []
                do_flip = random.random() < 0.5
                train_rand = np.random.rand()
                replay = None
                for frame_path in frame_path_list:
                    image = np.array(Image.open(frame_path).convert("RGB"))
                    landmark = np.load(
                        frame_path.replace(".png", ".npy").replace(
                            "/frames/", self.landmark_path
                        )
                    )[0]
                    bbox_landmark = np.array(
                        [
                            landmark[:, 0].min(),
                            landmark[:, 1].min(),
                            landmark[:, 0].max(),
                            landmark[:, 1].max(),
                        ]
                    )
                    bboxes = np.load(
                        frame_path.replace(".png", ".npy").replace(
                            "/frames/", "/retina/"
                        )
                    )[:2]
                    iou_max = -1
                    for i in range(len(bboxes)):
                        iou = IoUfrom2bboxes(bbox_landmark, bboxes[i].flatten())
                        if iou_max < iou:
                            iou_max = iou
                            bbox = bboxes[i]

                    landmark = self.reorder_landmark(landmark)
                    if self.phase == "train" and do_flip:
                        image, _, landmark, bbox = self.hflip(
                            image, None, landmark, bbox
                        )
                    image, landmark, bbox, _ = custom_crop_face(
                        image,
                        landmark,
                        bbox,
                        phase=self.phase,
                        train_rand=train_rand,
                        margin_factor=1/4,
                    )
                    original_height, original_width = image.shape[:2]
                    image = cv2.resize(
                        image,
                        (self.image_size, self.image_size),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    # Calculate the scaling factors for both height and width
                    height_scale = self.image_size / original_height
                    width_scale = self.image_size / original_width

                    # Apply the scaling factors to the landmarks
                    landmark = landmark.astype(float)
                    landmark[:, 0] *= width_scale
                    landmark[:, 1] *= height_scale

                    if replay:
                        transform = A.ReplayCompose.replay(replay, image=image)
                    else:
                        transform = self.transform(image=image)
                        replay = transform["replay"]

                    image = transform["image"]

                    # h, w, 3 -> 3, h, w
                    image = img_to_tensor(
                        image,
                        normalize={
                            "mean": IMAGENET_DEFAULT_MEAN,
                            "std": IMAGENET_DEFAULT_STD,
                        },
                    )

                    image_np = image.permute(1, 2, 0).numpy()
                    patches = self.extract_patches(image_np, landmark, self.patch_size)
                    patches = torch.from_numpy(patches).permute(0, 3, 1, 2)

                    image_list.append(image)
                    landmark_list.append(landmark)
                    patch_list.append(patches)
                flag = False

            except Exception as e:
                if self.verbose:
                    print(
                        f"[yellow]failed to load frame {frame_path} of video {video_name} due to {str(e)}[/yellow]"
                    )
                index = random.randint(0, len(self))

        image_list = torch.stack(image_list, axis=0)
        landmark_list = np.stack(landmark_list, axis=0)
        patch_list = torch.stack(patch_list, axis=0)
        return {
            "frames": image_list,
            "landmarks": landmark_list,
            "patches": patch_list,
            "label": self.label_dict[video_name],
        }

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate(
            [77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]
        ):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
        H, W = img.shape[:2]
        landmark = landmark.copy()
        bbox = bbox.copy()

        if landmark is not None:
            landmark_new = np.zeros_like(landmark)

            landmark_new[:17] = landmark[:17][::-1]
            landmark_new[17:27] = landmark[17:27][::-1]

            landmark_new[27:31] = landmark[27:31]
            landmark_new[31:36] = landmark[31:36][::-1]

            landmark_new[36:40] = landmark[42:46][::-1]
            landmark_new[40:42] = landmark[46:48][::-1]

            landmark_new[42:46] = landmark[36:40][::-1]
            landmark_new[46:48] = landmark[40:42][::-1]

            landmark_new[48:55] = landmark[48:55][::-1]
            landmark_new[55:60] = landmark[55:60][::-1]

            landmark_new[60:65] = landmark[60:65][::-1]
            landmark_new[65:68] = landmark[65:68][::-1]
            if len(landmark) == 68:
                pass
            elif len(landmark) == 81:
                landmark_new[68:81] = landmark[68:81][::-1]
            else:
                raise NotImplementedError
            landmark_new[:, 0] = W - landmark_new[:, 0]

        else:
            landmark_new = None

        if bbox is not None:
            bbox_new = np.zeros_like(bbox)
            bbox_new[0, 0] = bbox[1, 0]
            bbox_new[1, 0] = bbox[0, 0]
            bbox_new[:, 0] = W - bbox_new[:, 0]
            bbox_new[:, 1] = bbox[:, 1].copy()
            if len(bbox) > 2:
                bbox_new[2, 0] = W - bbox[3, 0]
                bbox_new[2, 1] = bbox[3, 1]
                bbox_new[3, 0] = W - bbox[2, 0]
                bbox_new[3, 1] = bbox[2, 1]
                bbox_new[4, 0] = W - bbox[4, 0]
                bbox_new[4, 1] = bbox[4, 1]
                bbox_new[5, 0] = W - bbox[6, 0]
                bbox_new[5, 1] = bbox[6, 1]
                bbox_new[6, 0] = W - bbox[5, 0]
                bbox_new[6, 1] = bbox[5, 1]
        else:
            bbox_new = None

        if mask is not None:
            mask = mask[:, ::-1]
        else:
            mask = None
        img = img[:, ::-1].copy()
        return img, mask, landmark_new, bbox_new

    def extract_patches(self, image, landmarks, patch_size=32):
        patches = []
        half_patch_size = patch_size // 2
        h, w = image.shape[:2]

        for landmark in landmarks:
            x, y = int(landmark[0]), int(landmark[1])
            x_min = max(0, x - half_patch_size)
            y_min = max(0, y - half_patch_size)
            x_max = min(w, x + half_patch_size)
            y_max = min(h, y + half_patch_size)

            patch = image[y_min:y_max, x_min:x_max]
            patch = cv2.resize(
                patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR
            )
            patches.append(patch)

        patches = np.stack(patches, axis=0)
        return patches


def visualize_cropped_faces_and_landmarks(frames, landmarks, out="face.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(frames), figsize=(20, 5))
    for i, (cropped_face, landmark) in enumerate(zip(frames, landmarks)):
        # Transpose the dimensions to (height, width, channels)

        cropped_face = cropped_face.permute(1, 2, 0).numpy()
        cropped_face = cropped_face * np.array(IMAGENET_DEFAULT_STD) + np.array(
            IMAGENET_DEFAULT_MEAN
        )
        axes[i].imshow(cropped_face)
        axes[i].scatter(landmark[:, 0], landmark[:, 1], c="r", s=5)
        axes[i].axis("off")
    plt.savefig(out)
    plt.close()


def visualize_patches(image, landmarks, patches, patch_size=32, out="patches.png"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.transpose(image, (1, 2, 0))
    image = (
        image * np.array(IMAGENET_DEFAULT_STD) + np.array(IMAGENET_DEFAULT_MEAN)
    ) * 255
    image = image.astype(np.uint8)

    num_landmarks = landmarks.shape[0]
    num_columns = 5
    num_rows = (num_landmarks + num_columns - 1) // num_columns

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 3 * num_rows))

    # Show the original image with landmarks
    ax = plt.subplot(num_rows, num_columns, 1)
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c="red", s=5)
    plt.title("Original Image with Landmarks")
    plt.axis("off")

    # Draw rectangles around the patches on the original image
    half_patch_size = patch_size // 2
    for lm in landmarks:
        rect = Rectangle(
            (lm[0] - half_patch_size, lm[1] - half_patch_size),
            patch_size,
            patch_size,
            linewidth=1,
            edgecolor="r",
            facecolor=(1, 0, 0, 0.1),
        )
        ax.add_patch(rect)

    # Show the patches
    for i, patch in enumerate(patches):
        ax = plt.subplot(num_rows, num_columns, i + 2)
        patch = np.transpose(
            patch, (1, 2, 0)
        )  # Transpose the patch to (height, width, channels)
        patch = (
            patch * np.array(IMAGENET_DEFAULT_STD) + np.array(IMAGENET_DEFAULT_MEAN)
        ) * 255
        patch = patch.astype(np.uint8)
        plt.imshow(patch)
        plt.title(f"Landmark {i + 1}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


if __name__ == "__main__":
    patch_size = 32
    dataset = FFPPVideoDataset(phase="train", image_size=336, patch_size=patch_size)
    for item in dataset:
        frames = item["frames"]
        landmarks = item["landmarks"]
        patches = item["patches"]
        visualize_cropped_faces_and_landmarks(frames, landmarks)
        visualize_patches(
            frames[0].numpy(), landmarks[0], patches[0].numpy(), patch_size=patch_size
        )
        break  # Remove this line if you want to visualize all videos in the dataset
