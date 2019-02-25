import torch
from fewshots.models.utils import to_variable


def make_float_label(true_samples, n_samples):
    label = torch.FloatTensor(n_samples).zero_()
    label[0:true_samples] = 1
    return to_variable(label)


def make_float_label_2d(true_samples, n_samples):
    label = torch.FloatTensor(n_samples, 1).zero_()
    label[0:true_samples] = 1
    return to_variable(label)


def make_byte_label(true_samples, n_samples):
    label = torch.ByteTensor(n_samples).zero_()
    label[0:true_samples] = 1
    return to_variable(label)
