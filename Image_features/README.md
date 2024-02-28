# Example
The script `run_cache_stim_grid_repr.py` is set to calculate and save stimulus features for an example session (Pa210120, 144 images at 16 × 16 dva) using default settings in the paper. The log file for this example is `./log_cache_stim*`.

# Workflow overview
The file `script_cache_stim_grid_repr.ipynb` contains the substantive code. This script calculates, for a list of images, model feature embeddings in these flavors:
- The full image;
- Image patches on a regular grid;
- A background patch filled with padding values (the default is uniform mean gray).

The `.py` code is a utility script to calculate image features for specific sessions. It prepares the per-session parameters (i.e., image list and presentation size) and then calls the `.ipynb` script.

# Implementation notes
## Image ID
The code expects each image to be identified by a unique filename 'stem' (i.e., without the extension) and looks for the corresponding image in the folder `[stim_dir]/Stimuli`. The images in the OSF repo are already organized this way, with the filename stem being the image's MD5 value (suffixes vary due to different image file formats). The motivation to use unique filenames is to avoid confusing distinct images with the same filename (in different folders).

Other code that uses the cached features (i.e., model-based analysis scripts) needs to identify images in the same way. The rest of the repo and data on DANDI are already organized thus.

## Same image shown across sessions
Most sessions show the same few image sets. Thus, the .py and .ipynb scripts both check for and skip images whose features have already been cached.
