# This file originally appeared in https://github.com/jakesnell/prototypical-networks
# and has been modified for the purpose of this project

import os
import json
import logging
from functools import partial

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

from fewshots.engine import Engine
import fewshots.utils.data as data_utils
import fewshots.utils.model as model_utils
import fewshots.utils.log as log_utils


def _data_model_setup(opts):
    if opts['data.dataset'] == 'omniglot':
        opts['data.split'] = 'vinyals'
        opts['model.x_dim'] = '1,28,28'
        if opts['model.model_name'] == 'RRNet':
            opts['model.out_dim'] = 1536 + 2048
        elif opts['model.model_name'] == 'RRNet_small':
            opts['model.out_dim'] = 256 * 2
        else:
            raise ValueError('Unknown model name')

    elif opts['data.dataset'] == 'miniimagenet':
        opts['data.split'] = 'ravi-larochelle'
        opts['model.x_dim'] = '3,84,84'
        if opts['model.model_name'] == 'RRNet':
            opts['model.out_dim'] = 31104 + 41472
        elif opts['model.model_name'] == 'RRNet_small':
            opts['model.out_dim'] = 5184 * 2
        else:
            raise ValueError('Unknown model name')

    elif opts['data.dataset'] == 'cifar100':
        opts['data.split'] = 'bertinetto'
        opts['model.x_dim'] = '3,32,32'
        if opts['model.model_name'] == 'RRNet':
            opts['model.out_dim'] = 3456 + 4608
        elif opts['model.model_name'] == 'RRNet_small':
            opts['model.out_dim'] = 576 * 2
        else:
            raise ValueError('Unknown model name')
    else:
        raise ValueError('Unknown dataset name')

    return opts


def main(opts):
    opts = _data_model_setup(opts)

    opts['model.x_dim'] = list(map(int, opts['model.x_dim'].split(',')))
    if not isinstance(opts['log.fields'], (list,)):
        opts['log.fields'] = opts['log.fields'].split(',')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts['data.gpu'])
    exp_folder = os.path.join('results', opts['log.exp_dir'])

    if not os.path.isdir(exp_folder):
        os.makedirs(exp_folder)

    # save opts
    with open(os.path.join(exp_folder, 'opt.json'), 'w') as f:
        json.dump(opts, f)
        f.write('\n')

    trace_file = os.path.join(exp_folder, 'trace.txt')
    log_file = os.path.join(exp_folder, 'log.txt')

    logger = setup_logger(log_file)

    torch.manual_seed(opts['data.seed'])
    if opts['data.cuda']:
        torch.cuda.manual_seed(opts['data.seed'])

    data = data_utils.load(opts, ['train', 'val1', 'val5'])
    train_loader = data['train']
    val1_loader = data['val1']
    val5_loader = data['val5']

    model = model_utils.load(opts)

    if opts['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = {'train': {field: tnt.meter.AverageValueMeter() for field in opts['log.fields']}}

    if val1_loader is not None and val5_loader is not None:
        meters['val1'] = {field: tnt.meter.AverageValueMeter() for field in opts['log.fields']}
        meters['val5'] = {field: tnt.meter.AverageValueMeter() for field in opts['log.fields']}

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        if opts['train.scheduler_type'] == 'step':
            state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opts['train.decay_every'],
                                                     gamma=opts['train.lr_decay'])
        elif opts['train.scheduler_type'] == 'plateau':
            state['scheduler'] = lr_scheduler.ReduceLROnPlateau(state['optimizer'], mode='max',
                                                                factor=opts['train.lr_decay'],
                                                                patience=opts['train.plateau_patience'],
                                                                threshold=0.002, threshold_mode='abs')
        else:
            raise ValueError('Unknown scheduler type.')

    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        if opts['train.scheduler_type'] == 'step':
            state['scheduler'].step()

    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])

    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val1_loader is not None:
            if 'best_acc1' not in hook_state:
                hook_state['best_acc1'] = 0
            if 'wait1' not in hook_state:
                hook_state['wait1'] = 0

        if val5_loader is not None:
            if 'best_acc5' not in hook_state:
                hook_state['best_acc5'] = 0
            if 'wait5' not in hook_state:
                hook_state['wait5'] = 0

        if val1_loader is not None:
            model_utils.evaluate(state['model'],
                                 val1_loader,
                                 meters['val1'],
                                 desc="  [1shot] Epoch {:d} valid".format(state['epoch']))

        if val5_loader is not None:
            model_utils.evaluate(state['model'],
                                 val5_loader,
                                 meters['val5'],
                                 desc="  [5shot] Epoch {:d} valid".format(state['epoch']))

        meter_vals = log_utils.extract_meter_values(meters)
        logger("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val1_loader is not None:
            if opts['train.scheduler_type'] == 'plateau':
                state['scheduler'].step(meter_vals['val1']['acc'])
            if meter_vals['val1']['acc'] >= hook_state['best_acc1']:
                hook_state['best_acc1'] = meter_vals['val1']['acc']
                logger("==> SAVING MODEL FOR 1SHOT")

                state['model'].cpu()
                torch.save(state['model'], os.path.join(exp_folder, 'best_model.1shot.t7'))
                if opts['data.cuda']:
                    state['model'].cuda()

                hook_state['wait1'] = 0
            else:
                hook_state['wait1'] += 1

        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(exp_folder, 'best_model.1shot.t7'))
            if opts['data.cuda']:
                state['model'].cuda()

        if val5_loader is not None:
            if opts['train.scheduler_type'] == 'plateau':
                state['scheduler'].step(meter_vals['val5']['acc'])
            if meter_vals['val5']['acc'] >= hook_state['best_acc5']:
                hook_state['best_acc5'] = meter_vals['val5']['acc']
                logger("==> SAVING MODEL FOR 5SHOT")

                state['model'].cpu()
                torch.save(state['model'], os.path.join(exp_folder, 'best_model.5shot.t7'))
                if opts['data.cuda']:
                    state['model'].cuda()

                hook_state['wait5'] = 0
            else:
                hook_state['wait5'] += 1

                if hook_state['wait1'] > opts['train.patience'] and hook_state['wait5'] > opts['train.patience']:
                    logger("==> patience {:d} exceeded".format(opts['train.patience']))
                    logger("==> BEST ACC for 1 and 5 shots: {:2.3f}, {:2.3f}".format(hook_state['best_acc1'] * 100,
                                                                                     hook_state['best_acc5'] * 100))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(exp_folder, 'best_model.5shot.t7'))
            if opts['data.cuda']:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, {})

    engine.train(
        model=model,
        loader=train_loader,
        optim_method=getattr(optim, opts['train.optim_method']),
        optim_config={'lr': opts['train.learning_rate'],
                      'weight_decay': opts['train.weight_decay']},
        max_epoch=opts['train.epochs']
    )


def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr
