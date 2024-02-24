import numpy as np

def get_patches_from_grid(iims, xy_degs, patch_grid_reprs, bg_repr, patch_bins_x, patch_bins_y):
    """
    iims: shape (n,) indexing into first axis of patch_grid_reprs
    xy_degs: shape (n, 2) to be digitized by patch_bins_{x,y}
    patch_grid_reprs: shape (n_im, n_patches_x, n_patches_y, *bg_repr.shape)
    bg_repr: any shape
    patch_bins_{x,y}: shape (n_patches_{x,y},)
    """
    feats = np.empty((len(iims), *bg_repr.shape), dtype=bg_repr.dtype)
    ixs = np.digitize(xy_degs[:,0], patch_bins_x)
    iys = np.digitize(xy_degs[:,1], patch_bins_y)
    m = (0 < ixs) & (ixs < len(patch_bins_x)) & (0 < iys) & (iys < len(patch_bins_y))
    feats[m] = patch_grid_reprs[iims[m], ixs[m]-1, iys[m]-1]
    feats[~m] = bg_repr  # fill out-of-bound locations
    return feats
