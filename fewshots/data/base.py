# This file originally appeared in https://github.com/jakesnell/prototypical-networks
# nd has been modified for the purpose of this project

import torch


def convert_dict(k, v):
    return {k: v}


class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k, v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data


class EpisodicBatchSampler(object):
    def __init__(self, SE, n_classes, n_episodes):
        self.setup_episode = SE
        self.n_classes = n_classes
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            n_way, _, _ = self.setup_episode.create_new_setup()
            yield torch.randperm(self.n_classes)[:n_way]