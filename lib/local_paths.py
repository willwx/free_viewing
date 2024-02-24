"""
Frequently used paths across many analyses are put here
Note, relative paths are relative to the main working dir (not necessarily this file location)
"""

from pathlib import Path
import json

__all__ = [
    'locale', 'data_root',
    'preproc_dir', 'analysis_dir', 'cache_dir', 'stim_dir',
    'project_dir', 'database_dir', 'annot_path',
    'eye_latency_calib_paths', 'similar_images_list_path',
    'mplstyle_path', 'papermill_bin']

with open(Path(__file__).parent / 'local_paths.json', 'r') as f:
    data = json.load(f)

locale = data.get('locale', 'local')

# location of data and analysis results
data_root = data.get('data_root', '~/Data/FreeViewing/')
preproc_dir = data.get('preproc_dir', data_root+'Preprocessed/')
analysis_dir = data.get('analysis_dir', data_root+'Analysis/')
cache_dir = data.get('cache_dir', data_root+'Cache/')
stim_dir = data.get('stim_dir', data_root+'Stimuli/')

# location of code and catalogs
project_dir = data.get('project_dir', '~/Documents/projects/Free Viewing/')
database_dir = data.get('database_dir', project_dir+'db/')
# - used for looking up the array id & region per bank, per session
annot_path = data.get('annot_path', database_dir+'bank_array_regions.csv')
# - used during preprocessing
eye_latency_calib_paths = data.get('eye_latency_calib_paths', (
    database_dir + 'sess_rig_info.csv',
    database_dir + 'rig_eye_latency.csv'))
# - used in modelling-related scripts
similar_images_list_path = data.get('similar_images_list_path', database_dir+'similar_images.txt')
# - used when plotting paper-sized figures
mplstyle_path = data.get('mplstyle_path', database_dir+'paper.mplstyle')

# path to papermill executable for batch running analysis as ipynbs
papermill_bin = data.get('papermill_bin', '~/.local/bin/papermill')
