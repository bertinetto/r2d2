import torch
from torch import nn
from torch.autograd import Variable
from torch import transpose as t
from torch import inverse as inv
from torch import mm
from torch import gesv
import numpy as np

from fewshots.labels_r2d2 import make_float_label, make_long_label
from fewshots.models.adjust import AdjustLayer, LambdaLayer
from fewshots.data.queries import shuffle_queries_multi

def t_(x):
    return t(x, 0, 1)


class RRNet(nn.Module):
    def __init__(self, encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base, n_augment, linsys):
        super(RRNet, self).__init__()
        self.encoder = encoder
        self.debug = debug
        self.lambda_rr = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.output_dim = out_dim
        self.n_augment = n_augment
        self.linsys = linsys

    def loss(self, sample):
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert(xs.size(0) == xq.size(0))
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1)
        if n_way * n_shot * self.n_augment > self.output_dim + 1:
            rr_type = 'standard'
            I = Variable(torch.eye(self.output_dim + 1).cuda())
        else:
            rr_type = 'woodbury'
            I = Variable(torch.eye(n_way * n_shot * self.n_augment).cuda())

        y_inner = make_float_label(n_way, n_shot * self.n_augment) / np.sqrt(n_way * n_shot * self.n_augment)
        y_outer_binary = make_float_label(n_way, n_query)
        y_outer = make_long_label(n_way, n_query)

        x = torch.cat([xs.view(n_way * n_shot*self.n_augment, *xs.size()[2:]),
                       xq.view(n_way * n_query, *xq.size()[2:])], 0)

        x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment, y_outer_binary, y_outer)

        z = self.encoder.forward(x)
        zs = z[:n_way * n_shot*self.n_augment]
        zq = z[n_way * n_shot*self.n_augment:]
        # add a column of ones for the bias
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        if rr_type == 'woodbury':
            wb = self.rr_woodbury(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        else:
            wb = self.rr_standard(torch.cat((zs, ones), 1), n_way, n_shot, I, y_inner, self.linsys)

        w = wb.narrow(dimension=0, start=0, length=self.output_dim)
        b = wb.narrow(dimension=0, start=self.output_dim, length=1)
        out = mm(zq, w) + b
        y_hat = self.adjust(out)
        # print("%.3f  %.3f  %.3f" % (w.mean()*1e5, b.mean()*1e5, y_hat.max()))

        _, ind_prediction = torch.max(y_hat, 1)
        _, ind_gt = torch.max(y_outer_binary, 1)

        loss_val = self.L(y_hat, y_outer)
        # how many of the query classes have been guessed correctly?
        acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
        # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
        return loss_val, {
            'loss': loss_val.data[0],
            'acc': acc_val.data[0]
        }

    def rr_standard(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way*n_shot*self.n_augment)

        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = gesv(v, A)

        return w

    def rr_woodbury(self, x, n_way, n_shot, I, yrr_binary, linsys):
        x /= np.sqrt(n_way * n_shot*self.n_augment)

        if not linsys:
            w = mm(mm(t(x, 0, 1), inv(mm(x, t(x, 0, 1)) + self.lambda_rr(I))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = gesv(v, A)
            w = mm(t_(x), w_)

        return w

