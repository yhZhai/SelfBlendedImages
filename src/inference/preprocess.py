import os
from pathlib import Path

from tqdm import tqdm
from rich import print
import numpy as np
import cv2
from PIL import Image
import torch


def extract_frames(filename, num_frames, model, image_size=(380, 380), extract_every_frame: bool = False):
    if extract_every_frame:
        print("[red]Extracting every frame, num_frames with be useless[/red]")

    cap_org = cv2.VideoCapture(filename)

    if not cap_org.isOpened():
        print(f"Cannot open: {filename}")
        return []

    croppedfaces = []
    idx_list = []
    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))

    if not extract_every_frame:
        frame_idxs = np.linspace(
            0, frame_count_org - 1, num_frames, endpoint=True, dtype=int
        )
    else:
        frame_idxs = np.arange(frame_count_org)

    for index in list(frame_idxs):
        cap_org.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret_org, frame_org = cap_org.read()
        if not ret_org:
            tqdm.write(
                "Frame read {} Error! : {}".format(
                    index, os.path.basename(filename)
                )
            )
            break

        frame = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

        faces = model.predict_jsons(frame)
        try:
            if len(faces) == 0:
                tqdm.write(
                    "No faces in {}:{}".format(index, os.path.basename(filename))
                )
                continue

            size_list = []
            croppedfaces_temp = []
            idx_list_temp = []

            for face_idx in range(len(faces)):
                x0, y0, x1, y1 = faces[face_idx]["bbox"]
                bbox = np.array([[x0, y0], [x1, y1]])
                croppedfaces_temp.append(
                    cv2.resize(
                        crop_face(
                            frame,
                            None,
                            bbox,
                            False,
                            crop_by_bbox=True,
                            only_img=True,
                            phase="test",
                        ),
                        dsize=image_size,
                    ).transpose((2, 0, 1))
                )
                idx_list_temp.append(index)
                size_list.append((x1 - x0) * (y1 - y0))

            max_size = max(size_list)
            croppedfaces_temp = [
                f
                for face_idx, f in enumerate(croppedfaces_temp)
                if size_list[face_idx] >= max_size / 2
            ]
            idx_list_temp = [
                f
                for face_idx, f in enumerate(idx_list_temp)
                if size_list[face_idx] >= max_size / 2
            ]
            croppedfaces += croppedfaces_temp
            idx_list += idx_list_temp
        except Exception as e:
            print(f"error in {index}:{filename}")
            print(e)
            continue
    cap_org.release()

    return croppedfaces, idx_list


def extract_face(frame, model, image_size=(380, 380)):

    faces = model.predict_jsons(frame)

    if len(faces) == 0:
        print("No face is detected")
        return []

    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]["bbox"]
        bbox = np.array([[x0, y0], [x1, y1]])
        croppedfaces.append(
            cv2.resize(
                crop_face(
                    frame,
                    None,
                    bbox,
                    False,
                    crop_by_bbox=True,
                    only_img=True,
                    phase="test",
                ),
                dsize=image_size,
            ).transpose((2, 0, 1))
        )

    return croppedfaces


def crop_face(
    img,
    landmark=None,
    bbox=None,
    margin=False,
    crop_by_bbox=True,
    abs_coord=False,
    only_img=False,
    phase="train",
):
    assert phase in ["train", "val", "test"]

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8  # 0#np.random.rand()*(w/8)
        w1_margin = w / 8
        h0_margin = h / 2  # 0#np.random.rand()*(h/5)
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == "train":
        w0_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        w1_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        h0_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
        h1_margin *= np.random.rand() * 0.6 + 0.2  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p, q) in enumerate(bbox):
            bbox_cropped[i] = [p - x0_new, q - y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return (
            img_cropped,
            landmark_cropped,
            bbox_cropped,
            (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
            y0_new,
            y1_new,
            x0_new,
            x1_new,
        )
    else:
        return (
            img_cropped,
            landmark_cropped,
            bbox_cropped,
            (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
        )


def extract_and_save_to_hdf5(video_list, target_list, target_path):
    import h5py
    from retinaface.pre_trained_models import get_model

    Path(target_path).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    face_detector = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    face_detector.eval()
    for filename in tqdm(video_list):
        face_list, idx_list = extract_frames(filename, 32, face_detector, extract_every_frame=True)
        video_name = Path(filename).stem
        save_path = Path(target_path, video_name + ".hdf5")
        with h5py.File(save_path, "w") as f:
            frame_dataset = f.create_dataset('frames', (len(face_list), *face_list[0].shape), dtype='uint8', compression="gzip", compression_opts=9)
            index_dataset = f.create_dataset('indices', (len(idx_list),), dtype='int64')

            for i in range(len(face_list)):
                frame_dataset[i] = face_list[i]
                index_dataset[i] = idx_list[i]


if __name__ == "__main__":
    
    from datasets import *
    video_list, target_list = init_cdf()

    extract_and_save_to_hdf5(video_list, target_list, ".cache/cdf")
