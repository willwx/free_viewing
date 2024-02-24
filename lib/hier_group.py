import re
import numpy as np
import pandas as pd


def unpack_hier_names(names):
    # standardize unit names into hierarchical format
    names_ = np.empty((len(names), 2), dtype=object)
    for i, n in enumerate(names):
        j = n.find('/')
        if j == -1:
            names_[i] = 'Unit', n
        else:
            names_[i] = n[:j], n[j+1:]
    return names_


def annot_names(names, annot_df):
    names_ = unpack_hier_names(names)
    m = names_[:,0] == 'Unit'
    assert set(names_[~m,0]) <= {'Channel', 'Bank', 'Array'}

    unit_df = pd.DataFrame(data={'Unit': names[m]})  # names_[m,1]
    unit_df['Channel'] = [int(re.search('\d+', v).group()) for v in names_[m,1]]
    unit_df['Bank'] = (unit_df['Channel'] - 1) // 32
    unit_df['Array ID'] = annot_df.loc[unit_df['Bank'].values, 'Array ID'].values
    unit_df['Level'] = 'Unit'

    unit_df = unit_df.set_index('Unit')
    for i in np.nonzero(~m)[0]:
        unit_df.loc[names[i]] = -1
        unit_df.loc[names[i], 'Level'] = names_[i,0]

    m = names_[:,0] == 'Channel'
    if m.any():
        unit_df.loc[names[m], 'Channel'] = chs = names_[m,1].astype(int)
        unit_df.loc[names[m], 'Bank'] = bks = (chs - 1) // 32
        unit_df.loc[names[m], 'Array ID'] = annot_df.loc[bks, 'Array ID'].values

    m = names_[:,0] == 'Bank'
    if m.any():
        unit_df.loc[names[m], 'Bank'] = bks = names_[m,1].astype(int)
        unit_df.loc[names[m], 'Array ID'] = annot_df.loc[bks, 'Array ID'].values

    m = names_[:,0] == 'Array ID'
    if m.any():
        unit_df.loc[names[m], 'Array ID'] = names_[m,1].astype(int)

    return unit_df.reset_index()


def hier_lookup(
        unit_names, annot_df, hier_df,
        hiers=('Unit', 'Channel', 'Bank', 'Array'),
        return_hier_indices=False):
    unit_df = annot_names(unit_names, annot_df)

    # look up RF fit per unit applying hierarchical fallback
    found = np.zeros(len(unit_df), dtype=bool)
    idcs = np.empty(len(unit_df), dtype=object)
    for i, (_, row) in enumerate(unit_df.iterrows()):
        for h in hiers:
            idx = (h, str(row[h.replace('Array', 'Array ID')]))
            if idx in hier_df.index:
                found[i] = True
                idcs[i] = idx
                break

    unit_df = unit_df[found]
    hier_df_ = hier_df.loc[idcs[found]].set_index(unit_df.index)
    unit_df = pd.concat([unit_df, hier_df_], axis=1)
    unit_df.index = hier_df_.index

    df = unit_df.set_index('Unit')
    if return_hier_indices:
        return df, idcs[found]
    return df
