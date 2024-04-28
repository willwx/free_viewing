from time import time
from typing import Optional, Union, Iterable

import numpy as np
import xarray as xr
from numba import njit, prange
from scipy.spatial.distance import pdist


@njit(fastmath=True, nogil=True, parallel=True)
def _pearsonr(x, y):
    """
    :param x: np.array shape (n_obs, n_feat)
    :param y: np.array shape (n_obs, n_feat)
    :return: np.array shape (n_feat,)
    """
    if len(x) < 2:
        return np.full(x.shape[1], np.nan, dtype=np.float32)
    r = np.empty(x.shape[1], dtype=np.float32)
    for i in prange(len(r)):
        xystd = np.std(x[:, i]) * np.std(y[:, i])
        if xystd:
            x_ = x[:, i] - np.mean(x[:, i])
            y_ = y[:, i] - np.mean(y[:, i])
            r[i] = np.mean(x_ * y_) / xystd
        else:
            r[i] = np.nan
    return r


@njit
def _pwsc_base(resps, pairs, flip):
    pairs[flip] = pairs[flip, ::-1]
    return _pearsonr(resps[pairs[:, 0]], resps[pairs[:, 1]])


@njit(nogil=True, parallel=True)
def _pwsc(resps, pairs, cent_idc, bs_idcs, perm_idcs, flips):
    n, c = resps.shape
    n_bs = len(bs_idcs)
    n_perm = len(perm_idcs)

    cent_pairs = pairs[cent_idc]
    cent = _pwsc_base(resps, cent_pairs, flips[0])

    bs = np.empty((n_bs, c), dtype=np.float32)
    for i in prange(n_bs):
        bs[i] = _pwsc_base(resps, pairs[bs_idcs[i]], flips[i + 1])

    perm = np.empty((n_perm, c), dtype=np.float32)
    p0, p1 = cent_pairs.T
    for i in prange(n_perm):
        pairs_ = np.empty_like(cent_pairs)
        pairs_[:, 0] = p0
        pairs_[:, 1] = p1[perm_idcs[i]]
        perm[i] = _pwsc_base(resps, pairs_, flips[i + n_bs + 1])

    return cent, bs, perm


def _pwsc_preamble(ndim, shape, coords, attrs):
    if coords is not None:
        assert len(coords) == ndim - 1
        for i, v in enumerate(coords):
            assert v[0] not in {'bootstrap', 'permutation'} \
                   and len(v[1]) == shape[1 + i]

    if attrs is not None:
        assert isinstance(attrs, dict)


def find_dtype_for_index(n):
    bit_depth = np.ceil(np.log2(n)).astype(int)
    assert bit_depth <= 64
    return (np.uint8, np.uint16, np.uint32, np.uint64) \
        [np.searchsorted((0, 8, 16, 32, 64), bit_depth) - 1]


def pairwise_self_consistency(
        resps: np.ndarray, pairs: np.ndarray,
        max_pairs: int = 10000, feats_batch_size: int = 65536,
        n_bootstraps: int = 100, n_permutations: int = 1000,
        alternatives: Union[str, Iterable] = 'greater',
        coords: Optional[tuple] = None, attrs: Optional[dict] = None,
        random_seed: Optional[int] = None, verbose: bool = False):
    if isinstance(alternatives, str):
        alternatives = (alternatives,)

    _pwsc_preamble(resps.ndim, resps.shape, coords, attrs)
    t0 = time()

    npairs = min(len(pairs), max_pairs)
    if verbose:
        print('calculating pairwise self consistency')
        print(
            f'\t{n_bootstraps} bootstraps, {n_permutations} null permutations')
        if len(pairs) > max_pairs:
            print(f'\tusing {max_pairs} pairs of {len(pairs)} ')
    if npairs < 2:
        raise ValueError(
            'Fewer than 2 pairs of return fixations; '
            'cannot calculate self consistency')

    resps_shape = resps.shape
    resps = resps.reshape(len(resps), -1)

    shape = pairs.shape
    uidc, pairs = np.unique(pairs, return_inverse=True)
    resps = resps[uidc]
    pairs = pairs.reshape(shape)

    rg = np.random.default_rng(random_seed)
    clip = lambda x: np.clip(x, -1, 1)

    dt = find_dtype_for_index(len(pairs))
    if npairs < len(pairs):
        cent_idc = rg.choice(len(pairs), size=npairs, replace=False).astype(dt)
    else:
        cent_idc = np.arange(len(pairs), dtype=dt)
    bs_idcs = rg.integers(0, len(pairs), size=(n_bootstraps, npairs), dtype=dt)
    perm_idcs = np.array(
        [rg.permutation(npairs) for _ in range(n_permutations)], dtype=dt)
    flips = rg.integers(2, size=(1 + n_bootstraps + n_permutations, npairs), dtype=bool)

    # calculate in batches because this can be memory intensive for very large resps
    n_feats = resps.shape[-1]
    n_batches = int(np.ceil(n_feats / feats_batch_size))
    cent, bs, perm = [], [], []
    if verbose and n_batches > 1:
        print(f'\tprocessing batch (total {n_batches}):', end='')

    for i_batch in range(n_batches):
        if verbose and n_batches > 1:
            if i_batch % 25 == 0: print('\n\t\t', end='')
            print(i_batch, end=' ')

        i0 = i_batch * feats_batch_size
        i1 = min(n_feats, (i_batch + 1) * feats_batch_size)

        cent_, bs_, perm_ = _pwsc(resps[:, i0:i1], pairs, cent_idc, bs_idcs, perm_idcs, flips)
        cent_, bs_, perm_ = map(clip, (cent_, bs_, perm_))  # clip to [-1, 1]; outside values possible due to fp rounding

        cent.append(cent_)
        bs.append(bs_)
        perm.append(perm_)

    if verbose and n_batches > 1: print()
    cent = np.concatenate(cent, axis=-1).reshape(resps_shape[1:])
    bs = np.concatenate(bs, axis=-1).reshape(n_bootstraps, *cent.shape)
    perm = np.concatenate(perm, axis=-1).reshape(n_permutations, *cent.shape)

    # make xarray
    if coords is None:
        coords = tuple(
            (f'dim {i}', np.arange(s, dtype=np.uint))
            for i, s in enumerate(resps_shape[1:]))
    dims = [v[0] for v in coords]
    coords = coords + (
        ('bootstrap', np.arange(n_bootstraps, dtype=np.uint)),
        ('permutation', np.arange(n_permutations, dtype=np.uint)))
    data_vars = {
        'sample': (dims, cent),
        'bootstraps': (['bootstrap'] + dims, bs),
        'permutations': (['permutation'] + dims, perm)}
    if attrs is None:
        attrs = {}
    attrs.update(dict(n_pairs=npairs))

    # turn permutation test into p-values
    if n_permutations > 0:
        ps = []
        for alt in alternatives:
            if alt == 'two-sided':
                d = np.abs(cent) - np.abs(perm)
            elif alt == 'greater':
                d = cent - perm
            else:
                d = perm - cent
            d = np.ma.masked_invalid(d)  # correlation can be nan
            p = 1 - (d > 0).mean(0)
            # permutation p is lower-bound by (valid) num perms
            p = np.clip(p, 1 / (1 + (~d.mask).sum(0)), None).filled(np.nan)
            ps.append(p)
        coords = coords + (('alternative', np.array(alternatives)),)
        data_vars['p-value'] = (['alternative'] + dims, np.array(ps))

    rtn = xr.Dataset(data_vars, coords=dict(coords), attrs=attrs)

    if verbose:
        print(f'done ({time() - t0:.1f} s)')
    return rtn


@njit(nogil=True, parallel=True)
def _pwsc_perm_pairs(resps, all_pairs, perm_idcs, n0, flips):
    n, c = resps.shape
    n_perm = len(perm_idcs)

    perm = np.empty((2, n_perm, c), dtype=np.float32)
    for i in prange(n_perm * 2):
        iperm = i // 2
        ipair = i % 2
        if ipair:
            s = slice(n0, None)
        else:
            s = slice(None, n0)
        pairs_ = all_pairs[perm_idcs[iperm, s]]
        flips_ = flips[iperm, s]
        perm[ipair, iperm] = _pwsc_base(resps, pairs_, flips_)
    return perm


def pairwise_self_consistency_perm_test(
        resps: np.ndarray, pairs0: np.ndarray, pairs1: np.ndarray,
        paired: bool = False, alternatives: Union[str, Iterable] = 'two-sided',
        max_pairs: int = 10000, feats_batch_size: int = 65536,
        n_permutations: int = 10000,
        coords: Optional[tuple] = None, attrs: Optional[dict] = None,
        random_seed: Optional[int] = None, verbose: bool = False):
    _pwsc_preamble(resps.ndim, resps.shape, coords, attrs)
    assert pairs0.ndim == pairs1.ndim == 2
    assert pairs0.shape[1] == pairs1.shape[1] == 2
    if paired:
        assert len(pairs0) == len(pairs1)
    assert max_pairs > 2
    if isinstance(alternatives, str):
        alternatives = (alternatives,)
    assert all(v in ('less', 'greater', 'two-sided') for v in alternatives)
    t0 = time()

    n0 = len(pairs0)
    n1 = len(pairs1)
    if n0 < 2 or n1 < 2:
        raise ValueError(
            'Fewer than 2 pairs of return fixations for one or more conditions '
            f'(n = {n0}, {n1}); '
            'cannot calculate self consistency')

    if verbose:
        print('calculating self consistency permutation distribution w.r.t. pairing conditions')
        print(f'{n_permutations} permutations')

    n = n0 + n1
    all_pairs = np.concatenate([pairs0, pairs1], axis=0)
    if n > 2 * max_pairs:
        n_ = 2 * max_pairs
        # subsample smaller condition with parabolic decay toward asymptote of
        # 4 pairs total, 2 pairs each cond; don't use linear decay because
        # that can heavily unbalance the two conditions
        na, nb = sorted((n0, n1))
        m = (max_pairs - 2) / (n / 2 - 2) ** 0.5
        n0_ = min(int(round(2 + m * (na - 2) ** .5)), na)
        n1_ = n_ - n0_
        if n0 > n1:
            n0_, n1_ = n1_, n0_
        if verbose:
            print(f'\tusing {n_} ({n0_}+{n1_}) pairs of {n} ({n0}+{n1})')
    else:
        n_ = n
        n0_ = n0
        n1_ = n1

    if paired:  # sanity check
        assert n0 == n1
        assert n0_ == n1_

    resps_shape = resps.shape
    resps = resps.reshape(len(resps), -1)

    rg = np.random.default_rng(random_seed)
    clip = lambda x: np.clip(x, -1, 1)

    dt = find_dtype_for_index(n)
    cent_idc = np.concatenate([
        rg.choice(n0, size=n0_, replace=False),
        n0 + rg.choice(n1, size=n1_, replace=False)
    ], axis=0)[None, :].astype(dt)
    if paired:
        perm_idcs = []
        for _ in range(n_permutations):
            idc0_ = rg.choice(n0, size=n0_, replace=False)
            iconds = rg.integers(0, 2, size=n0_)
            idc_ = np.array([idc0_, idc0_ + n0])
            idc_ = np.concatenate([
                idc_[iconds, range(n0_)],
                idc_[1-iconds, range(n0_)]])
            perm_idcs.append(idc_)
    else:
        perm_idcs = [
            rg.choice(n, size=n_, replace=False) for _ in range(n_permutations)]
    perm_idcs = np.array(perm_idcs, dtype=dt)
    flips = rg.integers(2, size=(1 + n_permutations, n_), dtype=bool)

    n_feats = resps.shape[-1]
    n_batches = int(np.ceil(n_feats / feats_batch_size))
    samp0, samp1, dsamp, dperm = [], [], [], []
    if verbose and n_batches > 1:
        print(f'\tprocessing batch (total {n_batches}):', end='')

    for i_batch in range(n_batches):
        if verbose and n_batches > 1:
            if i_batch % 25 == 0: print('\n\t\t', end='')
            print(i_batch, end=' ')

        i0 = i_batch * feats_batch_size
        i1 = min(n_feats, (i_batch + 1) * feats_batch_size)

        samp_ = _pwsc_perm_pairs(resps[:, i0:i1], all_pairs, cent_idc, n0_, flips[[0]])
        if n_permutations > 0:
            perm_ = _pwsc_perm_pairs(resps[:, i0:i1], all_pairs, perm_idcs, n0_, flips[1:])
        else:  # placeholder
            perm_ =- np.empty((2, 0, i1-i0), dtype=np.float32)
        samp_, perm_ = map(clip, (samp_, perm_))

        samp0.append(samp_[0, 0])
        samp1.append(samp_[1, 0])
        dsamp.append((samp_[1] - samp_[0])[0])
        dperm.append(perm_[1] - perm_[0])

    if verbose and n_batches > 1: print()
    samp0 = np.concatenate(samp0, axis=-1).reshape(resps_shape[1:])
    samp1 = np.concatenate(samp1, axis=-1).reshape(samp0.shape)
    dsamp = np.concatenate(dsamp, axis=-1).reshape(samp0.shape)
    dperm = np.concatenate(dperm, axis=-1).reshape(n_permutations, *samp0.shape)

    # prepare xr.Dataset
    if coords is None:
        coords = tuple(
            (f'dim {i}', np.arange(s, dtype=np.uint))
            for i, s in enumerate(resps_shape[1:]))
    dims = [v[0] for v in coords]
    coords = coords + (
        ('permutation', np.arange(n_permutations, dtype=np.uint)),)
    data_vars = {
        'cond0': (dims, samp0),
        'cond1': (dims, samp1),
        'diff': (dims, dsamp),
        'permuted_diffs': (['permutation'] + dims, dperm)}
    if attrs is None:
        attrs = {}
    attrs.update(dict(n_pairs0=n0_, n_pairs1=n1_, n_permutations=n_permutations))

    # turn permutation test into p-values
    ps = []
    for alt in alternatives:
        if alt == 'two-sided':
            d = np.abs(dsamp) - np.abs(dperm)
        elif alt == 'less':
            d = dsamp - dperm
        else:
            d = dperm - dsamp
        d = np.ma.masked_invalid(d)  # correlation can be nan
        p = 1 - (d > 0).mean(0)
        # permutation p is lower-bound by (valid) num perms
        p = np.clip(p, 1 / (1 + (~d.mask).sum(0)), None).filled(np.nan)
        ps.append(p)
    coords = coords + (
        ('alternative', np.array(alternatives)),)
    data_vars['p-value'] = (['alternative'] + dims, np.array(ps))

    # make xr.Dataset
    rtn = xr.Dataset(data_vars, coords=dict(coords), attrs=attrs)
    for k in ('cond0', 'cond1'):
        rtn[k].assign_attrs(dict(unit="Pearson's r"))
    for k in ('diff', 'permuted_diffs'):
        rtn[k].assign_attrs(dict(unit="Delta Pearson's r"))

    if verbose:
        print(f'done ({time() - t0:.1f} s)')
    return rtn


def find_return_fixations(
        imids: np.ndarray, rel_degs: np.ndarray, thres_deg: float = 1,
        times: Optional[np.ndarray] = None, thres_time: float = 10,
        distant: bool = False,
        verbose: bool = False):
    """
    :param imids: shape (n_fix,)
    :param rel_degs: shape (n_fix, 2)
    :param times: shape (n_fix,)
    """
    uids, inv = np.unique(imids, return_inverse=True)
    uid_idc = [[] for _ in range(len(uids))]
    for i, j in enumerate(inv):
        uid_idc[j].append(i)
    uid_idc = map(np.array, uid_idc)

    pairs = []
    npairs_per_im = np.empty(len(uids), dtype=np.uint)
    for i, idc in enumerate(uid_idc):
        if distant:
            m_return = pdist(rel_degs[idc, :]) > thres_deg  # shape (n_comb,)
        else:
            m_return = pdist(rel_degs[idc, :]) <= thres_deg  # shape (n_comb,)
        pw_idc = np.array(np.triu_indices(len(idc), k=1))
        idc0 = idc[pw_idc[0]]
        idc1 = idc[pw_idc[1]]
        if times is not None and thres_time:
            m_trial = np.abs(times[idc0] - times[idc1]) >= thres_time
            m_return &= m_trial
        pairs.append(np.array([idc0, idc1])[:, m_return])
        npairs_per_im[i] = pairs[-1].shape[1]
    try:
        pairs = np.concatenate(pairs, axis=-1).T  # shape (N, 2)
    except ValueError as e:
        if not 'need at least one array to concatenate' in str(e):
            raise
        pairs = np.empty((0, 2), dtype=np.uint64)
    if verbose:
        print(
            'return fixations:'
            f'\t{len(pairs)} pairs, {npairs_per_im.mean():.1f} '
            f'+/- {npairs_per_im.std():.1f} pairs per image')
    return pairs
