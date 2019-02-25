from PIL import Image
import torch
import numpy as np
from torchvision import transforms

normalize_mini_t = transforms.Normalize(mean=[x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]],
                                        std=[x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]])

normalize_cifar_t = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])


def nop(d):
    return d


def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d


def to_tensor(key, d):
    d[key] = transforms.functional.to_tensor(d[key])
    return d


def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[
        key].size[0], d[key].size[1])
    return d


def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d


def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d


def crop(key, crop_transforms, max_crop_shrink, d):
    transform_id = np.random.randint(max_crop_shrink)
    d[key] = crop_transforms[transform_id](d[key])
    return d


def normalize_mini_image(key, d):
    d[key] = normalize_mini_t(d[key])
    return d


def normalize_cifar_image(key, d):
    d[key] = normalize_cifar_t(d[key])
    return d
