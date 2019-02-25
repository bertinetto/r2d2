import torch
from fewshots.models.utils import to_variable


def make_float_label(n_way, n_samples):
    label = torch.FloatTensor(n_way*n_samples, n_way).zero_()
    for i in range(n_way):
        label[n_samples*i:n_samples*(i+1), i] = 1
    return to_variable(label)


def make_long_label(n_way, n_samples):
    label = torch.LongTensor(n_way*n_samples).zero_()
    for i in range(n_way*n_samples):
        label[i] = i // n_samples
    return to_variable(label)
