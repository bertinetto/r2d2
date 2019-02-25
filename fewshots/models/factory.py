# This file originally appeared in https://github.com/jakesnell/prototypical-networks and has been modified for the purpose of this project

MODEL_REGISTRY = {}


def register_model(model_name):
    def decorator(f):
        MODEL_REGISTRY[model_name] = f
        return f

    return decorator


def get_model(model_name, model_opt, data_opt, base_opt, augm_opt):
    if model_name in MODEL_REGISTRY:
        # merge option dictionaries
        opts = model_opt.copy()
        opts.update(data_opt)
        opts.update(base_opt)
        opts.update(augm_opt)
        return MODEL_REGISTRY[model_name](**opts)
    else:
        raise ValueError("Unknown model {:s}".format(model_name))
