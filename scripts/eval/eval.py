# This file originally appeared in https://github.com/jakesnell/prototypical-networks and has been modified for the purpose of this project

import os
import json
import math

import torch
import torchnet as tnt

from fewshots.utils import filter_opt, merge_dict
import fewshots.utils.data as data_utils
import fewshots.utils.model as model_utils


def main(opts):

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['data.gpu'])

    model = torch.load(opts['model.model_path'])
    if opts['data.cuda']:
        model.cuda()

    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opts['model.model_path']), 'opt.json')

    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # construct data
    data_opt = {'data.' + k: v for k, v in filter_opt(model_opt, 'data').items()}
    augm_opt = {'augm.' + k: v for k, v in filter_opt(model_opt, 'augm').items()}

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_episodes': 'data.train_episodes'
    }

    for k, v in episode_fields.items():
        if opts[k] != 0:
            data_opt[k] = opts[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    merged_opt = merge_dict(data_opt, augm_opt)

    print("Evaluating {:d}-way, {:d}-shot over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if opts['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load_data(merged_opt, ['test'])

    meters = {field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields']}

    model_utils.evaluate(model, data['test'], meters, desc="test")

    expm_path = opts['model.model_path'].split('/')
    expm_path = '/'.join(expm_path[:-1])
    fh = open(os.path.join(expm_path, 'eval_results.txt'), 'w')

    for field, meter in meters.items():
        mean, std = meter.value()
        results_str = "test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean,
                                                      1.96 * std / math.sqrt(data_opt['data.test_episodes']))
        print(results_str)
        fh.write(results_str+'\n')

    fh.close()
