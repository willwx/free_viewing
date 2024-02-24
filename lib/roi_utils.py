from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure


@lru_cache(maxsize=1024)
def get_circ_kernel(r):
    k = np.zeros((2 * r + 1, 2 * r + 1), np.uint8)
    cv2.circle(k, (r, r), r, 255, -1)
    return k


@lru_cache(maxsize=1024)
def load_roi_mask(imfn, cat, rois_dir, im_size_px, dil,
                  res=None, suffixes=('.png', '.jpg')):
    roidir = rois_dir / cat
    roi = None
    for s in suffixes:
        try:
            roi = Image.open(roidir / (Path(imfn).stem + s))
            break
        except FileNotFoundError:
            pass

    if roi is None:
        raise FileNotFoundError

    roi = roi.resize(im_size_px)
    roi = np.array(roi)
    if roi.ndim in (3,4):
        roi = roi[:,:,0]
    
    # roi should be a binary mask
    # round trip to bool fixes, e.g., jpg compression
    assert roi.dtype == np.uint8
    roi = (roi > 127).astype(np.uint8)  
    
    # how many non-contiguous ROIs per image?
    roi_cls = measure.label(roi)
    ucls = np.sort(np.unique(roi_cls))
    assert np.array_equal(ucls, np.arange(len(ucls)))
    
    if dil:
        kws = dict(
            pad_width=((dil, dil), (dil, dil)),
            mode='constant', 
            constant_values=0)
        roi = np.pad(roi, **kws).astype(np.uint8)
        roi_cls = np.pad(roi_cls, **kws).astype(np.uint16)
        if len(np.unique(roi_cls)) <= 2:
            dil_kern = get_circ_kernel(dil)
            roi = cv2.dilate(roi, dil_kern, iterations=1)
            roi_cls = cv2.dilate(roi_cls, dil_kern, iterations=1)
        else:
            roi0 = roi.copy()
            roi_cls0 = roi_cls.copy()
            if res is None: res = max(1, dil // 10)
            for dil_ in range(dil%res,dil+1,res):
                # do iterations manually; this is important for roi_cls
                # becuase cv2 dilate takes the maximum; see
                # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html
                if not dil_: continue
                dil_kern = get_circ_kernel(dil_)
                roi_new = cv2.dilate(roi0, dil_kern, iterations=1)
                roi_cls_new = cv2.dilate(roi_cls0, dil_kern, iterations=1)
                changed = (roi_new > 0) & (roi == 0)
                roi_cls[changed] = roi_cls_new[changed]
                roi = roi_new
        
    assert not (set(ucls.ravel()) - set(roi_cls.ravel()) - {0})
    
    roi = roi.astype(bool)
    return roi, roi_cls


@lru_cache(maxsize=16)
def read_csv_cached(path):
    return pd.read_csv(path, index_col=0)


@lru_cache(maxsize=1024)
def load_roi_bbox(imfn, cat, rois_dir, im_size_px, dil):
    bbox_df = read_csv_cached(rois_dir / f'{cat}.csv')
    try:
        vals = bbox_df.loc[imfn][['x','y','w','h','W','H']].values
    except KeyError:
        return None

    # how many bboxes per image?
    n_cls = vals.size // 6
    if n_cls != 1:
        raise NotImplementedError

    vals = vals.ravel()
    w0h0 = vals[-2:]
    xywh = vals[:4].reshape(2, 2)
    xywh = (im_size_px / w0h0) * xywh
    xywh[0] -= dil
    xywh[1] += xywh[0] + 2 * dil  # from xywh to x1y1x2y2
    return xywh


def tally_roi_cats(rois_dir):
    cats = []
    for d in rois_dir.iterdir():
        if d.is_dir() or d.suffix == '.csv':
            cats.append(d.stem)
    return cats


def is_on_roi(
        imfn: str,
        im_size_px: np.ndarray,
        rel_px: np.ndarray,
        cat: str,
        rois_dir: Path,
        dilate_px: int):
    """

    rel_px: shape (2,); rf location (fixation + rf center)
        relative to image center; (cartesian coords, so up is +)
    """
    dil = dilate_px
    
    # convert from image center-aligned, cartesian (up == (+))
    # to upper-left-corner aligned, CV conventional (up == (-))
    xy = rel_px * (1, -1) + im_size_px / 2
    xy = np.round(xy).astype(int)
    xy += dil  # imagine the image is padded on all sides by dil
    if np.any(xy < 0) or np.any(xy >= im_size_px + dil):
        return False, 0

    im_size_px_ = tuple(im_size_px)  # lru_cache does not like np arrays (unhashable)
    try:
        roi, roi_cls = load_roi_mask(imfn, cat, rois_dir, im_size_px_, dil)
        return roi[xy[1], xy[0]], roi_cls[xy[1], xy[0]]
    except FileNotFoundError:
        pass
    
    try:
        bbox = load_roi_bbox(imfn, cat, rois_dir, im_size_px_, dil)
        assert bbox is not None
        return np.all((bbox[0] <= xy) & (xy <= bbox[1])), 1
    except (AssertionError, FileNotFoundError):
        pass

    return False, 0
