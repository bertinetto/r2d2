# This file originally appeared in https://github.com/jakesnell/prototypical-networks and has been modified for the purpose of this project

import torch
from torch.autograd import Variable


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def roll(x, shift):
    return torch.cat((x[-shift:], x[:-shift]))


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)