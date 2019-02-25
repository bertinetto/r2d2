# This file originally appeared in https://github.com/jakesnell/prototypical-networks and has been modified for the purpose of this project

from fewshots.data.load import load_data


def load(opt, splits):
    print(opt['data.dataset'])

    ds = load_data(opt, splits)

    return ds
