import numpy as np
from statsmodels.stats.multitest import multipletests


def p2str(p, return_cmp=False):
    if p == 0:
        return '0'

    elif p < 1e-3:
        scirep = tuple(map(int, ('%.0e' % p).split('e')))
        if return_cmp:
            p_ = float('%de%d' % scirep)
            if p_ == p:
                return '= %.0fe%d' % scirep                
            elif p_ < p:
                scirep = (scirep[0]+1, scirep[1])
                if scirep[0] == 10:
                    scirep = (1, scirep[1]+1)
                    if scirep[1] == -3:
                        return '< 0.001'

            p_ = float('%de%d' % scirep)
            assert p_ >= p
            return '< %.0fe%d' % scirep
                
        else:
            return '%.0fe%d' % scirep

    else:
        return str(round(p, 3))


default_rg = np.random.default_rng(0)


def hier_center_boots(
        val_dict, stat_fun, n_bootstraps=1000, range_=range, rg=default_rg):
    """
    val_dict: val per root level; each val shape (ncond, ..., nelem_i)
    """
    stat_first = lambda v: stat_fun(v, axis=0)
    stat_last = lambda v: stat_fun(v, axis=-1)
    bootstrap_last = lambda v: v[..., rg.choice(v.shape[-1], v.shape[-1])]
    bs_stat_last = lambda v: stat_last(bootstrap_last(v))

    center = stat_first(list(map(stat_last, val_dict.values())))

    keys = np.array(list(val_dict.keys()))
    bs_center = np.array([
        stat_first([
            bs_stat_last(val_dict[k])  # bootstrap on the last axis
            for k in rg.choice(keys, keys.size)])  # bootstrap on the root level
        for _ in range_(n_bootstraps)])

    return center, bs_center


def hier_mean_boots(*args, **kwargs):
     return hier_center_boots(*args, stat_fun=np.nanmean, **kwargs)


def hier_perm_test(
        val_dict, test, alternative='both', conds=None,
        stat_fun=np.nanmean, n_permutation=10000, range_=range, rg=default_rg):
    """
    val_dict: val per root level; each val shape (ncond, ..., nelem_i)
    test: (cond0, cond1)
    conds: required unless cond0/1 are already integer indices
    """
    assert len(test) == 2
    i0, i1 = test
    if conds is not None:
        if not isinstance(conds, list): conds = list(conds)
        i0 = conds.index(i0)
        i1 = conds.index(i1)
    else:
        assert all(isinstance(i, int) for i in (i0, i1))

    permute_last = lambda v: v * rg.choice([-1,1], v.shape[-1])
    stat_first = lambda v: stat_fun(v, axis=0)
    stat_last = lambda v: stat_fun(v, axis=-1)
    perm_stat_last = lambda v: stat_last(permute_last(v))

    diff_dict = {
        k: v[i0] - v[i1]  # shape (nt, nu)
        for k, v in val_dict.items()}  # v shape (ncond, nt, nu)

    h1 = stat_first(list(map(stat_last, diff_dict.values())))

    h0s = np.array([  # null distribution
        stat_first(list(map(perm_stat_last, diff_dict.values())))
        for _ in range_(n_permutation)])
    # note that the above does not bootstrap on the root level,
    # which subtly affects the interpretation of the stat test

    h1 = np.ma.masked_invalid(h1)
    h0s = np.ma.masked_invalid(h0s)
    if alternative == 'greater':
        pvals = h0s >= h1
    elif alternative == 'less':
        pvals = h0s <= h1
    else:
        pvals = np.abs(h0s) >= np.abs(h1)
    isvalid = np.broadcast_to(~pvals.mask, pvals.shape)
    pvals = pvals.mean(0).filled(np.nan)
    pvals_floor = 1 / (isvalid.sum(0) + 1)
    return pvals, pvals_floor


def fdr_correction(pvals_dict, q=0.01, method='fdr_tsbky'):
    # put all p-vals into a long list
    ksvs = [[], [], []]
    for k, v in pvals_dict.items():
        ksvs[0].append(k)
        ksvs[1].append(v.shape)
        ksvs[2].append(v.ravel())

    if len(ksvs[2]):
        v = np.concatenate(ksvs[2])
        m = np.isfinite(v)
        reject = np.empty(v.size, dtype=bool)
        apvals = np.empty_like(v)
        if m.any():
            reject[m], apvals[m] = multipletests(v[m], alpha=q, method=method)[:2]
        if not m.all():
            reject[~m] = False
            apvals[~m] = np.nan
    else:
        reject = apvals = np.empty(0)

    # restore adjusted test results to original dict structure
    reject_dict = {}
    apvals_dict = {}
    split_idc = np.cumsum(list(map(np.prod, ksvs[1])))
    for k, s, r, a in zip(
            *ksvs[:2],
            np.array_split(reject, split_idc),
            np.array_split(apvals, split_idc)):
        reject_dict[k] = r.reshape(s)
        apvals_dict[k] = a.reshape(s)
    return reject_dict, apvals_dict


def get_bootstrap_spread(m, bs, spread_type, ci=(2.5,97.5), return_dev=False):
    assert spread_type in (
        'ci_of_mean', 'ci_of_median', 'boots_sem', 'boots_mad_of_median')
    if spread_type.startswith('ci'):
        assert not return_dev  # (symmetrical) deviations are undefined
        return np.nanpercentile(bs, ci, axis=0)
    else:
        if spread_type == 'boots_sem':
            s = np.sqrt(np.nanmean((bs - m) ** 2, 0))
        else:
            s = np.nanmedian(np.abs(bs - m), 0)
        if return_dev:
            return s
        return m + np.array([-s, s])


def corr_perm_test(xs, ys, alternative='greater', n_perm=10000, rg=default_rg):
    """ permutation test for Pearson's r """
    assert np.isfinite(xs).all() and np.isfinite(ys).all()
    r = np.corrcoef(xs, ys)[0, 1]
    perm_rs = np.empty(n_perm)
    ys = ys.copy()
    for i in range(n_perm):
        rg.shuffle(ys)
        perm_rs[i] = np.corrcoef(xs, ys)[0, 1]
    perm_rs = perm_rs[np.isfinite(perm_rs)]
    if alternative == 'greater':
        p = (perm_rs >= r).mean()
    elif alternative == 'less':
        p = (perm_rs <= r).mean()
    else:
        p = (np.abs(perm_rs) >= np.abs(r)).mean()
    p = np.clip(p, 1/(perm_rs.size+1), None)
    return p
