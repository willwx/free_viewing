from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GroupKFold

from local_paths import similar_images_list_path


def standardize(X):
    m = X.mean(0)
    s = X.std(0)
    s[s == 0] = 1
    return (X - m) / s


def pearsonr(a, b):
    try:
        if (not a.size) or np.isclose(a.var(), 0) or np.isclose(b.var(), 0):
            return np.nan
        return np.clip(np.corrcoef(a, b)[0, 1], -1, 1)
    except:
        return np.nan


@lru_cache(1)
def load_sim_im_map(path):
    sim_im_fn_map = {}
    with open(Path(path).expanduser(), 'r') as f:
        for l in f:
            vs = l.strip().split('\t')
            for v in vs[1:]:
                sim_im_fn_map[v] = vs[0]
    return sim_im_fn_map


def cv_split_by_image(
        imfns, n_splits, group_kfold=True,
        random_seed=None, verbose=True,
        sim_im_list_path=similar_images_list_path):

    if group_kfold:
        kf = GroupKFold(n_splits=n_splits)

        if sim_im_list_path is not None:
            if verbose:
                print('For group k-fold, loading catalog of similar images')
            sim_im_list_path = Path(sim_im_list_path).expanduser()
            assert sim_im_list_path.is_file()
            sim_im_map = load_sim_im_map(sim_im_list_path)
            # not using full (relative) path because even if different images
            # have the same name, it probably won't hurt to bundle them
            groups = np.array([sim_im_map.get(n, n) for n in imfns])
            if verbose:
                imfns = pd.Series(imfns)
                m = imfns.isin(sim_im_map)
                n_sim_rules = len(imfns[m].unique())
                print(f'accounting for {n_sim_rules} pairs of similar images')
        else:
            groups = imfns

        # manually shuffle since GroupKFold does not have this option
        o = np.arange(len(groups))
        np.random.default_rng(random_seed).shuffle(o)
        splits = list(kf.split(np.arange(len(groups)), groups=groups[o]))
        splits = [(o[i0], o[i1]) for i0, i1 in splits]

    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        splits = list(kf.split(imfns))

    train_mask = np.ones((n_splits, len(imfns)), dtype=bool)
    if verbose:
        print('num training and testing points per split:')
    for i, (i0, i1) in enumerate(splits):
        train_mask[i, i1] = False
        if verbose:
            print(f'split {i:d}\t\ttrain: {len(i0):5d} '
                  f'test: {len(imfns) - len(i0)}')

    return splits, train_mask


def cv_ridge_predict(X, Y, splits, alpha, return_pred=True, return_coefs=False):
    assert return_pred or return_coefs
    assert X.ndim == Y.ndim == 2
    if return_pred:
        cv_pred = np.zeros_like(Y)
    if return_coefs:
        coefs = np.zeros((len(splits), Y.shape[1], X.shape[1]), dtype=np.float32)

    for i_split, (i0, i1) in enumerate(splits):
        model = Ridge(alpha=alpha, max_iter=int(1e4))
        try:
            model.fit(X[i0], Y[i0])
        except LinAlgWarning as e:
            print(f'fitting failed at split {i_split}:', str(e))
            continue
        if return_pred:
            Y_pred = model.predict(X[i1])
            if Y_pred.ndim == 1: Y_pred = Y_pred[:,None]
            cv_pred[i1] = Y_pred
        if return_coefs:
            coefs[i_split] = model.coef_

    if return_pred:
        if return_coefs:
            return cv_pred, coefs
        return cv_pred
    elif return_coefs:
        return coefs


def cv_ridge_predict_eval(X, Y, splits, alpha):
    """
    Y: shape (n_samp, n_cond, n_dep_var)
    X: shape (n_samp, n_indep_var)
    splits: list of (n_samp_train, n_samp_test) indices
    """
    assert X.ndim == 2
    assert Y.ndim == 3
    assert len(Y) == len(X)
    n_win = Y.shape[1]
    n_split = len(splits)
    n_unit = Y.shape[2]

    corr_pers = np.full((n_win, n_split, n_unit), np.nan)
    r2_pers = np.full((n_win, n_split, n_unit), np.nan)
    corr = np.full((n_win, n_unit), np.nan)
    r2 = np.full((n_win, n_unit), np.nan)

    for iw in range(Y.shape[1]):
        Y_ = Y[:, iw]

        cv_pred = cv_ridge_predict(X, Y_, splits, alpha=alpha)

        for isp, (_, i1) in enumerate(splits):
            corr_pers[iw, isp] = np.array([pearsonr(y, p) for y, p in zip(Y_[i1].T, cv_pred[i1].T)])
            r2_pers[iw, isp] = r2_score(Y_[i1], cv_pred[i1], multioutput='raw_values')
        corr[iw] = np.array([pearsonr(y, p) for y, p in zip(Y_.T, cv_pred.T)])
        r2[iw] = r2_score(Y_, cv_pred, multioutput='raw_values')

    return corr, r2, corr_pers, r2_pers
