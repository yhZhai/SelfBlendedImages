import sys
import json
import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd
import albumentations as alb
import cv2


def load_json(path):
    d = {}
    with open(path, mode="r") as f:
        d = json.load(f)
    return d


def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def custom_crop_face(
    img,
    landmark=None,
    bbox=None,
    only_img=False,
    phase="train",
    train_rand=None,
    margin_factor=1/8,
):
    """
    Customizations:
    - crop_by_bbox always set to True.
    - margin always set to False.
    - abs_coord always set to False.


    This function crops a face from an input image, using either facial landmarks or a bounding box.
    
    Args:
        img (numpy.array): Input image.
        landmark (numpy.array, optional): Facial landmark coordinates.
        bbox (numpy.array, optional): Bounding box coordinates.
        margin (bool, optional): Whether to add a margin around the cropped face.
        abs_coord (bool, optional): Whether to return absolute coordinates of the cropped region.
        only_img (bool, optional): Whether to return only the cropped image.
        phase (str, optional): One of the following: "train", "val", "test". Determines the degree of randomness in cropping.

    Returns:
        img_cropped (numpy.array): Cropped face image.
        landmark_cropped (numpy.array, optional): Adjusted landmark coordinates.
        bbox_cropped (numpy.array, optional): Adjusted bounding box coordinates.
        (y0_new, y1_new, x0_new, x1_new) (tuple, optional): Absolute coordinates of the cropped region (if abs_coord=True).

    Raises:
        AssertionError: If neither landmark nor bbox is provided or if phase is not one of the allowed values.
    """

    assert phase in ["train", "val", "test"]
    assert landmark is not None or bbox is not None

    H, W = len(img), len(img[0])

    x0, y0 = bbox[0]
    x1, y1 = bbox[1]
    w = x1 - x0
    h = y1 - y0
    w0_margin = w * margin_factor
    w1_margin = w * margin_factor
    h0_margin = h * margin_factor
    h1_margin = h * margin_factor

    if phase == "train":
        if train_rand:
            w0_margin *= (train_rand + 0.5)
            w1_margin *= (train_rand + 0.5)
            h0_margin *= (train_rand + 0.5)
            h1_margin *= (train_rand + 0.5)
        else:
            w0_margin *= (np.random.rand() + 0.5)
            w1_margin *= (np.random.rand() + 0.5)
            h0_margin *= (np.random.rand() + 0.5)
            h1_margin *= (np.random.rand() + 0.5)

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin))
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin))

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
    return (
        img_cropped,
        landmark_cropped,
        bbox_cropped,
        (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1),
    )


def crop_face(
    img,
    landmark=None,
    bbox=None,
    margin=False,
    crop_by_bbox=True,
    abs_coord=False,
    only_img=False,
    phase="train",
    train_rand=None
):
    """
    This function crops a face from an input image, using either facial landmarks or a bounding box.
    
    Args:
        img (numpy.array): Input image.
        landmark (numpy.array, optional): Facial landmark coordinates.
        bbox (numpy.array, optional): Bounding box coordinates.
        margin (bool, optional): Whether to add a margin around the cropped face.
        crop_by_bbox (bool, optional): Whether to crop using the bounding box or landmarks.
        abs_coord (bool, optional): Whether to return absolute coordinates of the cropped region.
        only_img (bool, optional): Whether to return only the cropped image.
        phase (str, optional): One of the following: "train", "val", "test". Determines the degree of randomness in cropping.

    Returns:
        img_cropped (numpy.array): Cropped face image.
        landmark_cropped (numpy.array, optional): Adjusted landmark coordinates.
        bbox_cropped (numpy.array, optional): Adjusted bounding box coordinates.
        (y0_new, y1_new, x0_new, x1_new) (tuple, optional): Absolute coordinates of the cropped region (if abs_coord=True).

    Raises:
        AssertionError: If neither landmark nor bbox is provided or if phase is not one of the allowed values.
    """

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
        if train_rand:
            w0_margin *= train_rand * 0.6 + 0.2
            w1_margin *= train_rand * 0.6 + 0.2
            h0_margin *= train_rand * 0.6 + 0.2
            h1_margin *= train_rand * 0.6 + 0.2
        else:
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


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(
            img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST
        )
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds
