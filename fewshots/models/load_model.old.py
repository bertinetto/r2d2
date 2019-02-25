import torch
import torch.nn as nn

from fewshots.models import register_model
from fewshots.models import protonet, r2d2, lrd2, wide_resnet
from fewshots.utils import norm


def _norm(num_channels, bn_momentum, groupnorm=False):
    if groupnorm:
        return norm.GroupNorm(num_channels)
    else:
        return nn.BatchNorm2d(num_channels, momentum=bn_momentum)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class RRFeatures(nn.Module):
    def __init__(self, x_dim, parameters, lrelu_slope, drop, groupnorm, bn_momentum):
        super(RRFeatures, self).__init__()
        self.features1 = nn.Sequential(
                            nn.Conv2d(x_dim, parameters[0], 3, padding=1),
                            _norm(parameters[0], bn_momentum, groupnorm=groupnorm),
                            nn.MaxPool2d(2, stride=2),
                            nn.LeakyReLU(lrelu_slope))
        self.features2 = nn.Sequential(
                            nn.Conv2d(parameters[0], parameters[1], 3, padding=1),
                            _norm(parameters[1], bn_momentum, groupnorm=groupnorm),
                            nn.MaxPool2d(2, stride=2),
                            nn.LeakyReLU(lrelu_slope))
        self.features3 = nn.Sequential(
                            nn.Conv2d(parameters[1], parameters[2], 3, padding=1),
                            _norm(parameters[2], bn_momentum, groupnorm=groupnorm),
                            nn.MaxPool2d(2, stride=2),
                            nn.LeakyReLU(lrelu_slope),
                            nn.Dropout(drop))
        self.features4 = nn.Sequential(
                            nn.Conv2d(parameters[2], parameters[3], 3, padding=1),
                            _norm(parameters[3], bn_momentum, groupnorm=groupnorm),
                            nn.MaxPool2d(2, stride=1),
                            nn.LeakyReLU(lrelu_slope),
                            nn.Dropout(drop))

        self.pool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x3 = self.pool3(x)
        x3 = x3.view(x3.size(0), -1)
        x = self.features4(x)
        x4 = x.view(x.size(0), -1)
        x = torch.cat((x3, x4), 1)
        return x


@register_model('protonet_conv')
def load_protonet(**kwargs):

    dataset = kwargs['dataset']
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten()
    )

    # encoder = RRFeatures(x_dim[0], hid_dim, z_dim, dataset)

    return protonet.Protonet(encoder)


@register_model('WRN')
def load_wrn(**kwargs):
    drop = kwargs['drop']
    bn_momentum = kwargs['bn_momentum']
    out_dim = kwargs['out_dim']
    debug = kwargs['debug']
    learn_lambda = kwargs['learn_lambda']
    init_lambda = kwargs['init_lambda']
    init_adj_scale = kwargs['init_adj_scale']
    lambda_base = kwargs['lambda_base']
    adj_base = kwargs['adj_base']
    n_augment = kwargs['n_augment']
    linsys = kwargs['linsys']

    wrn_depth = kwargs['wrn_depth']
    wrn_width = kwargs['wrn_width']
    encoder = wide_resnet.WideResNet(wrn_depth, wrn_width, dropRate=drop)

    return r2d2.RRNet(encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys)


@register_model('RRNet')
def load_rrnet(**kwargs):
    lrelu = kwargs['lrelu']
    drop = kwargs['drop']
    groupnorm = kwargs['groupnorm']
    bn_momentum = kwargs['bn_momentum']
    out_dim = kwargs['out_dim']
    debug = kwargs['debug']
    learn_lambda = kwargs['learn_lambda']
    init_lambda = kwargs['init_lambda']
    init_adj_scale = kwargs['init_adj_scale']
    lambda_base = kwargs['lambda_base']
    adj_base = kwargs['adj_base']
    n_augment = kwargs['n_augment']
    linsys = kwargs['linsys']
    method = kwargs['method']
    iterations = kwargs['iterations']

    dataset = kwargs['dataset']
    if dataset == 'omniglot':
        x_dim = 1
    else:
        x_dim = 3

    parameters = [96, 192, 384, 512]
    encoder = RRFeatures(x_dim, parameters, lrelu, drop, groupnorm, bn_momentum)

    if method == 'R2D2':
        return r2d2.RRNet(encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys)
    else:
        return lrd2.LRD2(encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, iterations, linsys)


@register_model('RRNet_small')
def load_rrnet_small(**kwargs):
    lrelu = kwargs['lrelu']
    drop = kwargs['drop']
    groupnorm = kwargs['groupnorm']
    bn_momentum = kwargs['bn_momentum']
    out_dim = kwargs['out_dim']
    debug = kwargs['debug']
    learn_lambda = kwargs['learn_lambda']
    init_lambda = kwargs['init_lambda']
    init_adj_scale = kwargs['init_adj_scale']
    lambda_base = kwargs['lambda_base']
    adj_base = kwargs['adj_base']
    n_augment = kwargs['n_augment']
    linsys = kwargs['linsys']
    method = kwargs['method']
    iterations = kwargs['iterations']
    dataset = kwargs['dataset']
    if dataset == 'omniglot':
        x_dim = 1
    else:
        x_dim = 3

    parameters = [64, 64, 64, 64]
    encoder = RRFeatures(x_dim, parameters, lrelu, drop, groupnorm, bn_momentum)

    if method == 'R2D2':
        return r2d2.RRNet(encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys)
    else:
        return lrd2.LRD2(encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, iterations, linsys)
