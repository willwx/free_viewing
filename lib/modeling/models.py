import torchvision

try:
    import timm  # https://rwightman.github.io/pytorch-image-models/
except ModuleNotFoundError:
    timm = None
from torchvision import transforms
from .constants import preproc_scaling_opts
from .registry import get_default_preproc


def get_model_by_name(model_name, pretrained=True, dev=None):
    try:
        model = getattr(torchvision.models, model_name)(pretrained=pretrained)
        if dev is not None:
            model.to(dev)
        return model
    except AttributeError:
        pass

    if timm is None:
        raise ValueError(
            f'cannot find model with name {model_name}; however, timm is not '
            'installed and may provide the requested model; '
            'please visit https://rwightman.github.io/pytorch-image-models/ '
            'for installation instructions')

    try:
        model = timm.create_model(model_name, pretrained=pretrained)
        if dev is not None:
            model.to(dev)
        return model
    except AttributeError:
        raise ValueError(f'cannot find model with name {model_name}')


def get_preprocessor_by_model_name(
        model_name, preproc_imsize=None, preproc_scaling=None,
        preproc_from='image', return_config=False):
    assert preproc_from in ('array', 'numpy', 'tensor', 'pil', 'image')
    config = get_default_preproc(model_name)
    if preproc_imsize:
        config['imsize'] = preproc_imsize
    if preproc_scaling:
        config['scaling'] = preproc_scaling

    tforms = []
    if preproc_from not in ('pil', 'image'):
        tforms += [transforms.ToPILImage()]
    if preproc_imsize is not False:
        tforms += [
            transforms.Resize(config['imsize']),
            transforms.CenterCrop(config['imsize'])]
    tforms += [transforms.ToTensor()]
    if preproc_scaling is not False:
        transforms.Normalize(*preproc_scaling_opts[config['scaling']])

    preproc = transforms.Compose(tforms)

    if return_config:
        return preproc, config
    return preproc
