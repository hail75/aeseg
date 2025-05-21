import os
import random

import cv2
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


ImSurf = np.array([255, 255, 255])  # label 0
Building = np.array([255, 0, 0]) # label 1
LowVeg = np.array([255, 255, 0]) # label 2
Tree = np.array([0, 255, 0]) # label 3
Car = np.array([0, 255, 255]) # label 4
Clutter = np.array([0, 0, 255]) # label 5
Boundary = np.array([0, 0, 0]) # label 6
num_classes = 6


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)
    label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0
    label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
    return label_seg
