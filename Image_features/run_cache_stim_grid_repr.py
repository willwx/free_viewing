import sys
from pathlib import Path

# import numpy as np
import h5py as h5
import pandas as pd
import papermill as pm
from tqdm import tqdm

sys.path.append('../lib')
from local_paths import cache_dir, database_dir, preproc_dir
import json


#============================================================================
# Parameters
#============================================================================
# sessions = np.loadtxt(database_dir + '/sessions.txt', dtype=str)
sessions = ['Pa210120']

script_path = 'script_cache_stim_grid_repr.ipynb'

# these values are already defaults; see 'Parameter' cell in the ipynb
# defining here because the rest of this code uses some of these params
params = dict(
    patch_size=2,
    patch_step=0.5,
    model_name='vit_large_patch16_384',
    layer_name='blocks.13.attn.qkv',
    spatial_averaging=True,
    sep=',',
    device='cuda:0')


# ============================================================================
# Preamble
# ============================================================================
database_dir = Path(database_dir).expanduser()
preproc_dir = Path(preproc_dir).expanduser()
cache_dir = Path(cache_dir).expanduser()

model_name = params['model_name']
layer_name = params['layer_name']
output_dir = cache_dir / f'feats/{model_name}/{layer_name}'
log_dir = output_dir / 'log'
log_dir.mkdir(exist_ok=True, parents=True)

sep = params['sep']  # IFS for passing image list (MD5 values) to ipynb script
patch_size = params['patch_size']
patch_step = params['patch_step']


# ============================================================================
# Main
# ============================================================================
for sess in tqdm(sessions):
    proc_path = preproc_dir / (sess + '-proc.h5')

    # get image presentation size
    with h5.File(proc_path, 'r') as f:
        im_w, im_h = f['stimulus/size_dva'][()]
    im_w = round(im_w, 1)  # 0.1 dva res should be more than sufficient
    im_h = round(im_h, 1)
    suffix = f'{im_w:.1f}x{im_h:.1f}_as_{patch_size}x{patch_size}_in_{patch_step:.2f}_steps'
    output_path = output_dir / (suffix + '.h5')
    log_path = log_dir / (Path(script_path).stem + f'-{suffix}-{sess}.ipynb')

    # skip already completed session
    if log_path.is_file():
        try:
            nb = json.load(open(log_path, 'r'))
        except json.JSONDecodeError:
            nb = None
        if nb is not None:
            e = nb['metadata']['papermill']['exception']
            if not e:
                continue

    # tally images in this session
    fix_df = pd.read_hdf(proc_path, 'fixation_dataframe', 'r')
    # identify images by their MD5 (really, the filename stem, which needs to be
    # unique across images); uploaded images are named by their MD5
    md5s = set(Path(fn).stem for fn in fix_df['Image filename'].values)

    # skip if all images have been processed
    done_md5s = None
    if output_path.is_file():
        with h5.File(output_path, 'r') as f:
            try:
                done_md5s = f['md5'][()].astype(str)
                offset = len(done_md5s)
            except KeyError:
                pass
    if done_md5s is not None:
        rem_md5s = md5s - set(done_md5s)
        if not rem_md5s:
            continue

    # run job
    params.update(dict(
        im_md5s=sep.join(md5s),
        im_w=im_w,
        im_h=im_h))
    pm.execute_notebook(
        script_path,
        log_path,
        parameters=params,
        progress_bar=False)
