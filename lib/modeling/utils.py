import numpy as np
import torch

__all__ = [
    'make_layer_hook', 'recur_collapse_feats']


def _recur_clone(v):
    if isinstance(v, torch.Tensor):
        return v.clone()
    elif v is None:
        return None
    elif isinstance(v, tuple):
        return tuple(_recur_clone(v_) for v_ in v)
    elif isinstance(v, list):
        return [_recur_clone(v_) for v_ in v]
    else:
        raise ValueError(f'Cannot recursively clone input of type {type(v)}: {v}')


class ForwardHook:
    def __init__(self):
        self.i = None
        self.o = None

    def hook(self, module, i, o):
        self.i = i
        self.o = _recur_clone(o)


def make_layer_hook(model, layer_name, return_handle=False):
    hook = ForwardHook()
    hdl = None
    for layer_name_, layer in model.named_modules():
        if layer_name_ == layer_name:
            hdl = layer.register_forward_hook(hook.hook)
            break

    if return_handle:
        return hook, hdl
    return hook


def recur_collapse_feats(o, spatial_averaging=True):
    if isinstance(o, torch.Tensor):
        feats_ = o
        if spatial_averaging:
            if feats_.shape[0] == 1:  # first dim should match batch_size = 1
                feats_ = feats_[0]
            else:
                assert (np.array(feats_.shape) == 1).sum() == 1, \
                    f'unexpected feature shape {feats_.shape}'
                feats_ = feats_.squeeze()

            if feats_.ndim == 3:  # probably conv; avg space
                feats_ = feats_.mean((1, 2))
            elif feats_.ndim == 2:  # probably ViT; avg seq
                feats_ = feats_.mean(0)
            else:  # probably fc; do nothing
                assert feats_.ndim == 1, f'unexpected feature shape {feats_.shape}'
        else:
            feats_ = feats_.ravel()
        return feats_
    elif o is None:
        return None
    elif isinstance(o, tuple) or isinstance(o, list):
        vs = []
        for v in o:
            v = recur_collapse_feats(v)
            if v is not None:
                vs.append(v)
        return torch.cat(vs)
    else:
        raise ValueError(f'Cannot recursively collapse features of type {type(o)}: {o}')
