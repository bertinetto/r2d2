import os
import glob

from functools import partial

import torch

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
import torchvision
from fewshots.data import base, utils


def setup_images(split, d, cache, dataset, init_entry, root_dir, augm_opt):
    if dataset == 'omniglot':
        original_size = 105
        target_size = 28
        crop_transforms = [torchvision.transforms.RandomCrop(original_size-s) for s in range(augm_opt['max_crop_shrink'])]
        _setup_class_omniglot(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt)
    elif dataset == 'miniimagenet':
        original_size = 84
        target_size = 84
        crop_transforms = [torchvision.transforms.RandomCrop(original_size-s) for s in range(augm_opt['max_crop_shrink'])]
        _setup_class_miniimagenet(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt)
    elif dataset == 'cifar100':
        original_size = 32
        target_size = 32
        crop_transforms = [torchvision.transforms.RandomCrop(original_size-s) for s in range(augm_opt['max_crop_shrink'])]
        _setup_class_cifar100(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt)
    else:
        raise ValueError("Unknown dataset: {:s}".format(dataset))


def _setup_class_omniglot(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt):
    alphabet, character, rot = d['class'].split('/')
    image_dir = os.path.join(root_dir, 'omniglot', 'data', alphabet, character)

    if augm_opt['rotation']:
        rotation_f = partial(utils.rotate_image, 'data', float(rot[3:]))
    else:
        rotation_f = partial(utils.nop)
        print('WARNING - rotation augmentation is the default protocol for Omniglot')

    if augm_opt['crop']:
        crop_f = partial(utils.crop, 'data', crop_transforms, augm_opt['max_crop_shrink'])
    else:
        crop_f = partial(utils.nop)

    image_ds = TransformDataset(ListDataset(sorted(glob.glob(os.path.join(image_dir, '*.png')))),
                                compose([partial(base.convert_dict, 'file_name'),
                                         partial(utils.load_image_path, 'file_name', 'data'),
                                         rotation_f,
                                         crop_f,
                                         partial(utils.scale_image, 'data', target_size, target_size),
                                         partial(utils.convert_tensor, 'data'),
                                         ]))

    loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

    for sample in loader:
        if init_entry:
            cache.data[d['class']] = []

        cache.data[d['class']].append(sample['data'])
        break  # only need one sample because batch size equal to dataset length


def _setup_class_miniimagenet(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt):

    image_dir = os.path.join(root_dir, 'miniimagenet', 'data', d['class'])

    if augm_opt['rotation']:
        raise ValueError('Augmentation with rotation not implemented for miniimagenet')

    if augm_opt['crop']:
        crop_f = partial(utils.crop, 'data', crop_transforms, augm_opt['max_crop_shrink'])
        scale_f = partial(utils.scale_image, 'data', target_size, target_size)
    else:
        crop_f = partial(utils.nop)
        scale_f = partial(utils.nop)

    image_ds = TransformDataset(ListDataset(sorted(glob.glob(os.path.join(image_dir, '*.jpg')))),
                                compose([partial(base.convert_dict, 'file_name'),
                                         partial(utils.load_image_path, 'file_name', 'data'),
                                         crop_f,
                                         scale_f,
                                         partial(utils.to_tensor, 'data'),
                                         # partial(utils.normalize_mini_image, 'data')
                                         ]))

    loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

    for sample in loader:
        if init_entry:
            cache.data[d['class']] = []

        cache.data[d['class']].append(sample['data'])
        break  # only need one sample because batch size equal to dataset length


def _setup_class_cifar100(split, d, cache, init_entry, crop_transforms, target_size, root_dir, augm_opt):

    image_dir = os.path.join(root_dir, 'cifar100', 'data', d['class'])

    if augm_opt['rotation']:
        raise ValueError('Augmentation with rotation not implemented for cifar100')

    if augm_opt['crop']:
        crop_f = partial(utils.crop, 'data', crop_transforms, augm_opt['max_crop_shrink'])
        scale_f = partial(utils.scale_image, 'data', target_size, target_size)
    else:
        crop_f = partial(utils.nop)
        scale_f = partial(utils.nop)

    image_ds = TransformDataset(ListDataset(sorted(glob.glob(os.path.join(image_dir, '*.png')))),
                                compose([partial(base.convert_dict, 'file_name'),
                                         partial(utils.load_image_path, 'file_name', 'data'),
                                         crop_f,
                                         scale_f,
                                         partial(utils.to_tensor, 'data'),
                                         # partial(utils.normalize_cifar_image, 'data')
                                         ]))

    loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

    for sample in loader:
        if init_entry:
            cache.data[d['class']] = []

        cache.data[d['class']].append(sample['data'])
        break  # only need one sample because batch size equal to dataset length