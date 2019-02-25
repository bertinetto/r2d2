import torch
from fewshots.models.utils import to_variable


def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples).zero_()
    label[0:n_way * n_samples // 2] = 1
    return to_variable(label)


def make_float_label_2d(n_way, n_samples):
    label = torch.FloatTensor(n_way * n_samples, 1).zero_()
    label[0:n_way * n_samples // 2] = 1
    return to_variable(label)


def make_byte_label(n_way, n_samples):
    label = torch.ByteTensor(n_way * n_samples).zero_()
    label[0:n_way * n_samples // 2] = 1
    return to_variable(label)
