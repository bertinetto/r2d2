import torch
from torch import nn
from torch.autograd import Variable
from torch import transpose as t
from torch import inverse as inv
from torch import mm
from torch import gesv
from fewshots import labels_lrd2_multi, labels_lrd2_bin, labels_r2d2
from fewshots.models.adjust import AdjustLayer, LambdaLayer
from fewshots.models.utils import roll
from fewshots.data.queries import shuffle_queries_multi, shuffle_queries_bin


class LRD2(nn.Module):
    def __init__(self, encoder, debug, out_dim, learn_lambda, init_lambda, init_adj_scale, lambda_base, adj_base,
                 n_augment, irls_iterations, linsys):
        super(LRD2, self).__init__()
        self.encoder = encoder
        self.debug = debug
        self.lambda_ = LambdaLayer(learn_lambda, init_lambda, lambda_base)
        self.L = nn.CrossEntropyLoss()
        self.L_bin = nn.BCEWithLogitsLoss()
        self.adjust = AdjustLayer(init_scale=init_adj_scale, base=adj_base)
        self.output_dim = out_dim
        self.n_augment = n_augment
        assert (irls_iterations > 0)
        self.iterations = irls_iterations
        self.linsys = linsys

    def loss(self, sample):
        xs, xq = Variable(sample['xs']), Variable(sample['xq'])
        assert (xs.size(0) == xq.size(0))
        n_way, n_shot, n_query = xs.size(0), xs.size(1), xq.size(1)
        x = torch.cat([xs.view(n_way * n_shot * self.n_augment, *xs.size()[2:]),
                       xq.view(n_way * n_query, *xq.size()[2:])], 0)

        if n_way > 2:
            # 1-vs-all for multi-class
            y_inner_binary = labels_lrd2_multi.make_float_label(n_shot, n_way * n_shot * self.n_augment)
            y_outer_binary = labels_r2d2.make_float_label(n_way, n_query)
            y_outer = labels_r2d2.make_long_label(n_way, n_query)
            x, y_outer_binary, y_outer = shuffle_queries_multi(x, n_way, n_shot, n_query, self.n_augment,
                                                               y_outer_binary, y_outer)
            zs, zq = self.encode(x, n_way, n_shot)
            # save n_way scores per query, pick best for each query to know which class it is
            scores = Variable(torch.FloatTensor(n_query * n_way, n_way).zero_().cuda())
            for i in range(n_way):
                # re-init weight
                w0 = Variable(torch.FloatTensor(n_way * n_shot * self.n_augment).zero_().cuda())
                wb = self.ir_logistic(zs, w0, y_inner_binary)
                y_hat = mm(zq, wb)
                # y_hat = self.adjust(out)
                scores[:, i] = y_hat
                # re-generate base-learner label by circ-shift of n_shot steps
                y_inner_binary = roll(y_inner_binary, n_shot)

            _, ind_prediction = torch.max(scores, 1)
            _, ind_gt = torch.max(y_outer_binary, 1)

            loss_val = self.L(scores, y_outer)
            acc_val = torch.eq(ind_prediction, ind_gt).float().mean()
            # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
            return loss_val, {
                'loss': loss_val.data[0],
                'acc': acc_val.data[0]
            }

        else:
            y_inner_binary = labels_lrd2_bin.make_float_label(n_way, n_shot * self.n_augment)
            y_outer = labels_lrd2_bin.make_byte_label(n_way, n_query)
            y_outer_2d = labels_lrd2_bin.make_float_label(n_way, n_query).unsqueeze(1)
            x, y_outer, y_outer_2d = shuffle_queries_bin(x, n_way, n_shot, n_query, self.n_augment, y_outer, y_outer_2d)

            zs, zq = self.encode(x, n_way, n_shot)

            w0 = Variable(torch.FloatTensor(n_way * n_shot * self.n_augment).zero_().cuda())
            wb = self.ir_logistic(zs, w0, y_inner_binary)
            y_hat = mm(zq, wb)
            # y_hat = self.adjust(out)
            ind_prediction = (torch.sigmoid(y_hat) >= 0.5).squeeze(1)

            loss_val = self.L_bin(y_hat, y_outer_2d)
            acc_val = torch.eq(ind_prediction, y_outer).float().mean()
            # print('Loss: %.3f Acc: %.3f' % (loss_val.data[0], acc_val.data[0]))
            return loss_val, {
                'loss': loss_val.data[0],
                'acc': acc_val.data[0]
            }

    def encode(self, X, n_way, n_shot):
        z = self.encoder.forward(X)
        zs = z[:n_way * n_shot * self.n_augment]
        zq = z[n_way * n_shot * self.n_augment:]
        ones = Variable(torch.unsqueeze(torch.ones(zs.size(0)).cuda(), 1))
        zs = torch.cat((zs, ones), 1)
        ones = Variable(torch.unsqueeze(torch.ones(zq.size(0)).cuda(), 1))
        zq = torch.cat((zq, ones), 1)
        return zs, zq

    def ir_logistic(self, X, w0, y_inner):
        # iteration 0
        eta = w0  # + zeros
        mu = torch.sigmoid(eta)
        s = mu * (1 - mu)
        z = eta + (y_inner - mu) / s
        S = torch.diag(s)
        # Woodbury with regularization
        w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_(inv(S))))
        z_ = t(z.unsqueeze(0), 0, 1)
        w = mm(w_, z_)
        # it 1...N
        for i in range(self.iterations - 1):
            eta = w0 + mm(X, w).squeeze(1)
            mu = torch.sigmoid(eta)
            s = mu * (1 - mu)
            z = eta + (y_inner - mu) / s
            S = torch.diag(s)
            z_ = t(z.unsqueeze(0), 0, 1)
            if not self.linsys:
                w_ = mm(t(X, 0, 1), inv(mm(X, t(X, 0, 1)) + self.lambda_(inv(S))))
                w = mm(w_, z_)
            else:
                A = mm(X, t(X, 0, 1)) + self.lambda_(inv(S))
                w_, _ = gesv(z_, A)
                w = mm(t(X, 0, 1), w_)

        return w
