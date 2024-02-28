Code for the analyses in Xiao et al., 2024, 'Out of sight, out of mind: Feature-selective responses in the macaque follow eye movements during natural vision'. See this [early version on biorxiv](https://www.biorxiv.org/content/10.1101/2023.02.08.527666v1.full). A revised version is AIP at Nature Neuroscience.

The associated data are in [this DANDI repository](https://dandiarchive.org/dandiset/000628) (including neural and behavioral data) and this OSF repository (including images; upcoming).


# Examples
The folder `test_results` contains analysis code and results for an example session; the corresponding data are provided in `test_data` for convenience. The results include:
- Jupyter notebooks (.ipynb), one per analysis, that contain code and example plots;
- HDF5 (.h5) files that contain numerical results.

The subfolders are for runs with alternative parameters (e.g., using a 200 ms response window instead of sliding 50 ms windows).

The results are produced by the bash script, `run_one_sess_all_main.sh`. The script provides both a demo and further documentation of analysis parameters and dependencies.


# Repo structure
1. `Analysis`: Parameterized scripts (.ipynb) for per-session analyses.

2. `lib`: Code for frequently used subroutines. 

3. `db`: Parameters for analyses and plotting, including image region labels (masks and bounding boxes), lists of neurons (all, visually responsive, and visually selective), and per-neuron parameters (latencies and RFs; more details below).

4. `Image_features`: Scripts to pre-calculate DNN image embeddings used in model-based analyses (Figs. 5 & 6). 

Upcoming:
1. `Summary`: Scripts for statistical tests and summary plots with embedded results matching the paper figures.

2. `Postprocessing`: Scripts to extract data-estimated parameters used in downstream analyses. See [Workflow overview](#Workflow-overview) for details.


# To repduce the analyses
The script `run_one_sess_all_main.sh` can be applied to all sessions. The following notes further explain the pipeline and setup.

## Workflow overview
Each session is analyzed independently*. For each analysis and session, the same script is run using the session name as a parameter, This is done in batch using [papermill](papermill.readthedocs.io). The results are saved in .h5 files for use in summary plots and, sometimes, as inputs to downstream analyses.

*Some analyses use per-neuron paramters (latencies and RFs) estimated from data across sessions. This procedure is detailed in the paper Methods and the script `run_one_sess_hier_group.sh`. The folder `db` contains these parameters pre-calculated to facilitate reproducibility.

(Upcoming.) The code in `Image_features` is run once to pre-calculate image DNN embeddings (used in scripts 6a, 6b, 7a). This facilitates analyzing hundreds of sessions in parallel without needing GPUs. (Note that the cached features need ~200 GB of disk space using the default settings in the paper.)


## File locations
See `lib/local_paths.[py,json]` for details on where to put files (preprocessed neural data and images) and where outputs will be saved by default. The analysis scripts assume all preprocessed data are in `[data_root]/Preprocessed` (no subfolders) and named like the files in `test_data`. Files downloaded from DANDI will have a different naming/folder-structure convention and need to be reorganized.


## Requirements
Reasonably recent versions of Python and the following libraries.
Generally used libraries include numpy, h5py, pandas, scipy, scikit-learn, numba, xarray, matplotlib, seaborn, and tqdm.
Specific analyses and/or preprocessing use Pillow (PIL), scikit-image,
opencv-python, and pytorch.
We use [papermill](https://papermill.readthedocs.io/en/latest/) to batch-run (and log the results of) analyses over many sessions.