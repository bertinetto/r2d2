# This file originally appeared in https://github.com/jakesnell/prototypical-networks
# and has been modified for the purpose of this project

from tqdm import tqdm

from fewshots.utils import filter_opt
from fewshots.models import get_model


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']
    data_opt = filter_opt(opt, 'data')
    base_opt = filter_opt(opt, 'base_learner')
    augm_opt = filter_opt(opt, 'augm')

    del model_opt['model_name']

    return get_model(model_name, model_opt, data_opt, base_opt, augm_opt)


def evaluate(model, data_loader, meters, desc=None):
    model.eval()

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:
        _, output = model.loss(sample)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
