# based on timm
overall_default_preproc = {'imsize': 224, 'scaling': 'imagenet'}
default_preproc = {
    'vit_large_patch16_224': {'scaling': 'inception'},
    'vit_large_patch16_384': {'imsize': 384, 'scaling': 'inception'}}

def get_default_preproc(model_name):
    c = overall_default_preproc.copy()
    c.update(default_preproc.get(model_name, {}))
    return c
