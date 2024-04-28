import numpy as np


def axis_off_save_labels(ax):
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor('none')
    return ax


def plot_stat_sig(ax, xs, ms, y0, y1, **fill_kwargs):
    y0 = np.array(y0)
    y1 = np.array(y1)
    assert xs.ndim == ms.ndim == 1
    assert xs.size == ms.size
    assert y0.size == y1.size == 1
    assert ms.dtype == np.bool_
    flips = np.diff(np.concatenate([[0], ms.astype(int), [0]]))
    i0s = np.nonzero(flips > 0)[0]
    i1s = np.nonzero(flips < 0)[0][:len(i0s)]
    for i0, i1 in zip(i0s, i1s):
        i0, i1 = np.clip([i0, i1], 0, len(xs) - 1)
        ax.fill_between(xs[[i0, i1]], [y0, y0], [y1, y1], **fill_kwargs)


def get_color_kw(which, c, ec=False):
    assert which in ('ec', 'fc')
    if which == 'ec':
        return dict(ec=c, fc='none')
    if ec:
        return dict(ec=c, fc=c)
    return dict(ec='none', fc=c)


def plot_region_tests(
        xs,
        axs,
        regions,
        summary,
        palette,
        ybound,
        indicate_null=0,
        cond_open=None,
        xfrac=1,
        yfrac=0.05,
        yskip=0,
        lw=1,
        ec=False):
    '''
    axs: list of matplotlib axes, one per region
    regions: regions for axs
    ybound: range of y values;
        significance indications will be plotted above ybound[1]
        in steps of yfrac * (ybound[1] - ybound[0])
    '''
    byregion_center_boots = summary['one-level_center_boots']
    stat_tests = summary['stat_tests']
    reject_dict = summary['reject_dict']
    sharey = not isinstance(ybound, dict)

    kws = dict(step='mid', clip_on=False, lw=lw)
    if cond_open is None:
        cond_open = {}
    else:
        kws.update(dict(ec='none'))

    conds_ = list(summary['conds'])
    ymax = 0 if sharey else {}
    ybound_ = ybound
    for iax, (region, ax) in enumerate(zip(regions, axs)):
        if isinstance(ybound, dict):
            ybound_ = ybound[region]
        dy = (ybound_[1] - ybound_[0]) * yfrac
        y_ = y_0 = ybound_[1] + dy
        cond1 = None

        last_test = None
        for i, test in enumerate((*stat_tests, None)):
            if test is not None:
                last_test = test
                m = reject_dict[(region, test)]
                if test[2] == 'two-sided':
                    mu = byregion_center_boots[region][0]
                    d = mu[conds_.index(test[1])] - mu[conds_.index(test[0])]
                    for j in range(2):
                        m_ = m & ((d < 0) ^ bool(j))
                        color_which = ('fc', 'ec')[cond_open.get(test[j], False)]
                        kws.update(get_color_kw(color_which, palette[test[j]], ec))
                        plot_stat_sig(ax, xs, m_, y_, y_ + dy, **kws)
                else:
                    j = 1 - int(test[2] == 'greater')
                    color_which = ('fc', 'ec')[cond_open.get(test[j], False)]
                    kws.update(get_color_kw(color_which, palette[test[j]], ec))
                    plot_stat_sig(ax, xs, m, y_, y_ + dy, **kws)

            if indicate_null and (iax == 0 or indicate_null > 1):
                test = last_test
                if (cond1 is not None and cond1 != test[1]) or i == len(stat_tests):
                    if not cond1: cond1 = test[1]
                    color_which = ('fc', 'ec')[cond_open.get(cond1, True)]
                    kws_ = get_color_kw(color_which, palette[cond1], ec)
                    ax.fill_betweenx(
                        (y_0, y_ - dy * yskip), xs[0],
                        xs[0] - xfrac * (xs[1] - xs[0]), clip_on=False, **kws_)
                    cond1 = test[1]
                    y_0 = y_
                if cond1 is None:
                    cond1 = test[1]

            y_ += dy * (1 + yskip)

        if sharey:
            ymax = max(ymax, y_)
        else:
            ymax[region] = y_
    return ymax


def annotate_per_region_axes(
        axs,
        regions,
        ns_per_region,
        conds,
        palette,
        region_labels=None,
        show_num_subj=True,
        h=0.9,
        r=1,
        d=0.1):
    if region_labels is None:
        region_labels = regions

    for region, label, ax in zip(regions, region_labels, axs):
        ns = ns_per_region[region].values()
        ns = np.array(list(ns))
        l = f'n = {ns.sum()}'
        if label:
            l = f'{label}\n' + l
        if show_num_subj:
            l += f'/{ns.size}'
        ax.text(.05, h, l, ha='left', va='top', transform=ax.transAxes, fontsize=6)

    ax = axs[0]
    for cond in conds:
        ax.text(
            r, h, cond,
            ha='right', va='top', transform=ax.transAxes,
            color=palette[cond], fontsize=6)
        h -= d
