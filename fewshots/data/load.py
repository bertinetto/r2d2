import os
import numpy as np
import torch
from functools import partial
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset

from fewshots.data.base import convert_dict, CudaTransform, EpisodicBatchSampler
from fewshots.data.setup import setup_images
from fewshots.data.cache import Cache
from fewshots.utils import filter_opt
from fewshots.data.SetupEpisode import SetupEpisode

root_dir = ''


def extract_episode(setup_episode, augm_opt, d):
    # data: N x C x H x W
    n_max_examples = d[0]['data'].size(0)

    n_way, n_shot, n_query = setup_episode.get_current_setup()

    example_inds = torch.randperm(n_max_examples)[:(n_shot + n_query)]

    support_inds = example_inds[:n_shot]
    query_inds = example_inds[n_shot:]

    xs_list = [d[i]['data'][support_inds] for i in range(augm_opt['n_augment'])]
    # concatenate as shots into xs
    xs = torch.cat(xs_list, dim=0)
    # extract queries from a single cache entry
    xq = d[np.random.randint(augm_opt['n_augment'])]['data'][query_inds]
    out_dict = {'class': d[0]['class'], 'xs': xs, 'xq': xq, 'n_way': n_way, 'n_shot': n_shot, 'n_query': n_query}
    return out_dict


def load_data(opt, splits):
    global root_dir
    root_dir = opt['data.root_dir']
    augm_opt = filter_opt(opt, 'augm')
    dataset = opt['data.dataset']
    split_dir = os.path.join(opt['data.root_dir'], opt['data.dataset'], 'splits', opt['data.split'])

    ret = {}
    # cache = {}
    cache = Cache()

    for split in splits:
        if split in ['val1', 'val5', 'test']:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['train', 'trainval']:
            # random shots
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'],
                              fixed_shot=opt['data.shot'], way_min=opt['data.way_min'], fixed_way=n_way)
        elif split == 'val1':
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'], fixed_shot=1,
                              way_min=opt['data.way_min'], fixed_way=n_way)
        elif split == 'val5':
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'], fixed_shot=5,
                              way_min=opt['data.way_min'], fixed_way=n_way)
        else:
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'],
                              fixed_shot=opt['data.test_shot'], way_min=opt['data.way_min'], fixed_way=n_way)

        if split in ['val1', 'val5', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_images, split, dataset, cache, augm_opt),
                      partial(extract_episode, SE, augm_opt)]

        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        split_file = 'val.txt' if split in ['val1', 'val5'] else "{:s}.txt".format(split)
        with open(os.path.join(split_dir, split_file), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)

        sampler = EpisodicBatchSampler(SE, len(ds), n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret


def load_class_images(split, dataset, cache, augm_opt, d):
    if d['class'] in cache.data.keys():
        if len(cache.data[d['class']]) < augm_opt['cache_size']:
            init_entry = False
            setup_images(split, d, cache, dataset, init_entry, root_dir, augm_opt)
    else:
        init_entry = True
        setup_images(split, d, cache, dataset, init_entry, root_dir, augm_opt)

    cache_len = len(cache.data[d['class']])

    # if cache does not enough shots yet, repeat
    if cache_len < augm_opt['n_augment']:
        rand_ids = np.random.choice(cache_len, size=augm_opt['n_augment'], replace=True)
    else:
        rand_ids = np.random.choice(cache_len, size=augm_opt['n_augment'], replace=False)

    out_dicts = [{'class': d['class'], 'data': cache.data[d['class']][rand_ids[i]]} for i in
                 range(augm_opt['n_augment'])]

    return out_dicts
