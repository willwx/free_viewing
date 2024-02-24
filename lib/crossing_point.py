import numpy as np
import numba


def get_crossing_indices(y0, y1, direction='both'):
    """
    y0, y1: signals that together broadcast to shape (nsamp,)
    direction:
        up = y1 crosses y0 from below to above;
        down = y1 crosses y0 from above to below;
        or both
    """
    assert direction in ('both', 'up', 'down')
    y0, y1 = np.broadcast_arrays(y0, y1)
    assert y0.ndim == y1.ndim == 1
    assert y0.size == y1.size

    flip = np.diff((y1 > y0).astype(int))
    if direction == 'up':
        flip = flip > 0
    elif direction == 'down':
        flip = flip < 0
    return np.nonzero(flip)[0] + 1


@numba.njit(nogil=True, fastmath=True)
def interp_crossing_index_to_point(x1, y1, x2, y2, x):
    """
    x1, x2: coordinates of shapes (nsamp1,) (nsamp2,)
    y1, y2: signals of shapes (nsamp1,) (nsamp2,)
    x: scalar coordinate; must be within range of both x1 and of x2
    """
    assert x1.ndim == x2.ndim == y1.ndim == y2.ndim == 1
    assert x1.size == y1.size
    assert x2.size == y2.size
    x = np.array(x)

    i = np.digitize(x, x1)
    x11, x12 = x1[i - 1:i + 1]
    y11, y12 = y1[i - 1:i + 1]
    dx1 = x12 - x11
    dy1 = y12 - y11

    i = np.digitize(x, x2)
    x21, x22 = x2[i - 1:i + 1]
    y21, y22 = y2[i - 1:i + 1]
    dx2 = x22 - x21
    dy2 = y22 - y21

    b = dy1 * dx2 - dy2 * dx1
    if b == 0:
        return np.nan
    a = (y21 - y11) * dx1 * dx2 + x11 * dy1 * dx2 - x21 * dy2 * dx1
    return a / b


def get_crossing_points(x, y0, y1, direction):
    """
    x: coordinates of shape (nsamp,)
    y0, y1: signals, both of shape (nsamp,) (or implicitly so broadcast)
    """
    idc = get_crossing_indices(y0, y1, direction)
    if not idc.size:
        return None
    xstep = np.diff(x)
    cps = np.empty(idc.size)
    m0 = np.isfinite(y0)
    y0 = y0[m0]
    m1 = np.isfinite(y1)
    y1 = y1[m1]
    if isinstance(y0, np.ma.masked_array):
        y0 = y0.compressed()
    if isinstance(y1, np.ma.masked_array):
        y1 = y1.compressed()
    for i, idx in enumerate(idc):
        x_ = x[idx] - xstep[idx - 1] / 2
        cps[i] = interp_crossing_index_to_point(x[m0], y0, x[m1], y1, x_)
    return cps


def get_central_crossing_point_and_clearance(x, y0, y1, x_cent=0, direction='up'):
    assert x.ndim == y0.ndim == y1.ndim == 1
    assert x.size == y0.size == y1.size

    m = np.array(np.isfinite(y0) & np.isfinite(y1))
    if m.sum() < 2:
        return np.nan, np.nan, None
    
    x = x[m]
    y0 = np.array(y0[m])  # convert possibly masked array to array
    y1 = np.array(y1[m])
    t0s = get_crossing_points(x, y0, y1, direction=direction)

    if t0s is None:
        t0 = cl = np.nan
    else:
        t0s = np.unique(t0s)
        if t0s.size == 1:
            t0 = t0s.item()
            cl = np.min(np.abs(x[[0, -1]] - t0))
        else:
            t0 = t0s[np.argmin(np.abs(t0s - x_cent))]
            cl = np.abs(t0s - t0)
            cl = np.min(cl[cl != 0])

    return t0, cl, {'all crossings': t0s}


def hier_get_crossing_point(ts, y0s, y1s, hiers, t0=0, direction='up', clearance_thres=100):
    """
    ts:       (n_t,)
    y0s, y1s: (n_t, n_neur)
    hiers:    (n_hier, n_neur), first rows are broader levels in hierarchy
    """
    n_neur = y0s.shape[1]
    n_hier = len(hiers)
    if n_hier:
        h0 = hiers[0]
    else:
        h0 = np.arange(n_neur)
    assert h0.size == y0s.shape[1] == y1s.shape[1]

    uh0 = sorted(set(h0))
    hcps = np.empty((n_hier + 1, n_neur), dtype=np.float32)
    hcls = np.empty((n_hier + 1, n_neur), dtype=np.float32)

    for h in uh0:
        m = h0 == h
        idc = np.nonzero(m)[0]
        if not n_hier:
            # make sure the lowest hierarchy contains only leaf nodes
            assert len(idc) == 1

        y0 = y0s[:, m].mean(1)
        y1 = y1s[:, m].mean(1)
        cp, cl = get_central_crossing_point_and_clearance(
            ts, y0, y1, x_cent=t0, direction=direction)[:2]
        # note that all levels (dim 0) are assigned, but levels 1+
        # are overwritten below, if n_hier and (len(idc) > 1)
        hcps[:, idc] = cp
        hcls[:, idc] = cl

        if n_hier and (len(idc) > 1):
            # recursively go down hierarchy
            t0_ = cp if cl >= clearance_thres else t0
            hcps[1:, idc], hcls[1:, idc] = hier_get_crossing_point(
                ts, y0s[:, m], y1s[:, m], hiers=[h[m] for h in hiers[1:]],
                t0=t0_, direction=direction, clearance_thres=clearance_thres)

    return hcps, hcls


def hier_get_peak_time(ts, ys, hiers):
    """
    ts:     (n_t,)
    ys:     (n_t, n_neur)
    hiers:  (n_hier, n_neur), first rows are broader levels in hierarchy
    """
    n_neur = ys.shape[1]
    n_hier = len(hiers)
    if n_hier:
        h0 = hiers[0]
    else:
        h0 = np.arange(n_neur)
    assert h0.size == ys.shape[1]

    uh0 = sorted(set(h0))
    hpks = np.empty((n_hier + 1, n_neur), dtype=ys.dtype)
    htps = np.empty((n_hier + 1, n_neur), dtype=np.float32)

    for h in uh0:
        m = h0 == h
        idc = np.nonzero(m)[0]
        y = ys[:, m].mean(1)

        pk = y.max()
        if np.ma.isMA(pk):
            pk = pk.filled(np.nan)
        hpks[:, idc] = pk

        if np.isnan(pk):
            tp = np.nan
        else:
            tp = ts[np.nonzero(y == pk)[0][0]]
        htps[:, idc] = tp

        if n_hier and (len(idc) > 1):
            hpks[1:, idc], htps[1:, idc] = hier_get_peak_time(
                ts, ys[:, m], hiers=[h[m] for h in hiers[1:]])

    return hpks, htps