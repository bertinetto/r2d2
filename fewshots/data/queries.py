import torch
from torch.autograd import Variable


def shuffle_queries_bin(x, n_way, n_shot, n_query, n_augment, y_outer, y_outer_2d):
    ind_xs = torch.linspace(0, n_way * n_shot * n_augment - 1, steps=n_way * n_shot * n_augment).long()
    ind_xs = Variable(ind_xs.cuda())
    perm_xq = torch.randperm(n_way * n_query).long()
    perm_xq = Variable(perm_xq.cuda())
    permute = torch.cat([ind_xs, perm_xq + len(ind_xs)])
    return x[permute, :, :, :], y_outer[perm_xq], y_outer_2d[perm_xq]


def shuffle_queries_multi(x, n_way, n_shot, n_query, n_augment, y_binary, y):
    ind_xs = torch.linspace(0, n_way * n_shot * n_augment - 1, steps=n_way * n_shot * n_augment).long()
    ind_xs = Variable(ind_xs.cuda())
    perm_xq = torch.randperm(n_way * n_query).long()
    perm_xq = Variable(perm_xq.cuda())
    permute = torch.cat([ind_xs, perm_xq + len(ind_xs)])
    x = x[permute, :, :, :]
    y_binary = y_binary[perm_xq, :]
    y = y[perm_xq]
    return x, y_binary, y
