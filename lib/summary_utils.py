import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from local_paths import annot_path
from stats_utils import hier_center_boots, hier_perm_test, fdr_correction

known_regions = ['V1', 'V2', 'V4', 'PIT', 'CIT', 'AIT']
region_palette = {k: v for k, v in zip(known_regions, plt.get_cmap('inferno_r')(
    np.linspace(0.15, 0.8, len(known_regions))))}


def verified_update(ori, new):
    for k, v in new.items():
        try:
            assert np.array_equal(ori[k], v)
        except KeyError:
            ori[k] = v
    return ori


def annotate_unit_df(unit_df, annot_path=annot_path):
    unit_df['Subject'] = [v[:2] for v in unit_df['Session']]
    unit_df['Bank'] = [
        (int(re.search('\d+', n).group()) - 1) // 32
        for n in unit_df['Unit']]

    unit_df = unit_df.set_index(['Session', 'Bank'])
    adf = pd.read_csv(annot_path).set_index(['Session', 'Bank'])
    unit_df = pd.concat([unit_df, adf.reindex(unit_df.index)], axis=1)
    return unit_df


def select_units(
        unit_df,
        selection_path=None,
        select_valid_values=None,
        exclude_rare_subjects_per_region=0.05,
        return_regions=True):
    """
    unit_df: pd.DataFrame
        n_unit rows and columns [Session, Unit, Subject, Region]
    selection_path: file to csv file with columns [Session, Unit]
    select_valid_values: one of the following
        - None (not using)
        - np array of boolean type and shape (n_unit,)
            these units will be kept
        - np array of shape (..., n_unit);
            all values in all but last dimension must be valid
    exclude_rare_subjects_per_region:
        0–1, find median num units per subject, per region
        exclude subjects per region contributing fewer than 
        this fraction of median units
    """
    
    m = np.ones(len(unit_df), dtype=bool)
    m0 = m.copy()

    # Select based on file (e.g., visually selective units)
    if selection_path is not None:
        unit_sel_path = Path(selection_path)
        usel_idc = pd.MultiIndex.from_frame(
            pd.read_csv(unit_sel_path)[['Session', 'Unit']])
        unit_idc = pd.MultiIndex.from_frame(
            unit_df.reset_index()[['Session','Unit']])
        m &= unit_idc.isin(usel_idc)
        print(f'> Selected units: n = {m.sum()} of {m.size} '
              f'({m.mean()*100:.1f}% of data, '
              f'{usel_idc.isin(unit_idc).mean()*100:.1f}% of selection) '
              f'based on {unit_sel_path.name}')
        m0 = m.copy()
    
    # Select valid values
    if select_valid_values is not None:
        if select_valid_values.dtype == np.bool_:
            assert select_valid_values.shape == m.shape
            m &= select_valid_values
        else:
            assert select_valid_values.shape[-1] == m.size
            m &= np.isfinite(select_valid_values).reshape(-1, m.size).all(0)
        print(f'> Selected units: n = {m.sum()} of {m0.sum()} '
              f'({m[m0].mean()*100:.1f}%) with valid values')
        m0 = m.copy()
    
    # Exclude rare subjects per region
    if exclude_rare_subjects_per_region:
        assert 'Region' in unit_df.columns and 'Subject' in unit_df.columns
        cts = unit_df[m].groupby(['Region', 'Subject'])[['Unit']].count()
        med_cts = cts.groupby('Region').median().loc[
            cts.reset_index()['Region'].values, 'Unit'].values
        cts['Norm. units'] = cts['Unit'].values / med_cts
        excl_reg_subjs_ = cts[cts['Norm. units']<exclude_rare_subjects_per_region]
        excl_reg_subjs = excl_reg_subjs_.index
        if not len(excl_reg_subjs):
            str_ = str(cts.groupby('Region').min()['Norm. units'])
            print(f'> No rare subject to exclude')
            print('\t min norm. units per region:', '\n\t'.join(str_.split('\n')))
        else:
            m &= ~pd.Series([tuple(v) for v in unit_df[['Region', 'Subject']].values])\
                .isin(excl_reg_subjs.values).values
            str_ = str(excl_reg_subjs_.rename(columns={'Unit':'Units'}))
            print(f'> Excluded {(len(excl_reg_subjs))} rare subjects per region:')
            print('\t'+'\n\t'.join(str_.split('\n')))
            print(f'  Selected units: n = {m.sum()} of {m0.sum()} '
                  f'({m[m0].mean()*100:.1f}%)')
    
    unit_df = unit_df[m].copy()
    unit_df['Index'] = np.arange(len(unit_df))
    usel = np.nonzero(m)[0]
    if not return_regions:
        return unit_df, usel
    return unit_df, usel, tally_regions(unit_df)


def tally_regions(unit_df):
    return sorted(set(unit_df['Region'])-{None}, key=known_regions.index)


def summarize_results_per_region(
        unit_df,
        result_vals,
        level1='Region',
        level2='Subject',
        conds=None,
        spread_type='ci_of_mean',
        n_bootstraps=1000,  # for spread of center estimate
        stat_tests=None,
        n_permutation=10000,  # for stat test
        fdr_level=0.01,
        random_seed=0):
    """
    unit_df: pd.DataFrame
        n_unit rows and columns [Region, Subject, Index]
    result_vals: np array of shape (n_cond, n_t, n_unit)
        stat_test is done between condition pairs (on 1st dim)
        entries along n_t are independent (except when controlling for FDR)
    conds: list or tuple: names of conditions along n_cond dimension
    spread_type: one of 'ci_of_mean', 'ci_of_median', 'boots_sem', 'boots_mad_of_median'
        used to choose the center estimate, nanmean or nanmedian
    stat_tests: list or tuple of (cond1, cond2, alternative)
    """
    assert len(unit_df) == result_vals.shape[-1]
    assert spread_type in (
        'ci_of_mean', 'ci_of_median', 'boots_sem', 'boots_mad_of_median')
    if stat_tests is not None:
        assert conds is not None
        assert len(conds) == len(result_vals)
    rg = np.random.default_rng(random_seed)
    unit_df = unit_df.copy()
    unit_df['Index'] = np.arange(len(unit_df))

    lv1_lv2_vals = {}
    lv1_lv2_nunit = {}
    pvals_dict = {} if stat_tests is not None else None

    stat_fun = np.nanmedian if 'of_median' in spread_type else np.nanmean

    for lv1, df1 in unit_df.groupby(level1):
        lv1_lv2_vals[lv1] = lv2_vals = {}
        lv1_lv2_nunit[lv1] = lv2_nunit = {}

        for lv2, df2, in df1.groupby(level2):
            idc = df2['Index'].values
            lv2_vals[lv2] = result_vals[..., idc]
            lv2_nunit[lv2] = idc.size
        
        if stat_tests is not None:
            for test in stat_tests:
                perm_p, perm_p_floor = hier_perm_test(
                    lv2_vals, test[:2], test[2],
                    conds=conds, n_permutation=n_permutation, rg=rg)
                pvals_dict[(lv1, test)] = np.clip(perm_p, perm_p_floor, None)

    if pvals_dict is None:
        reject_dict = apvals_dict = None
    else:
        reject_dict, apvals_dict = fdr_correction(pvals_dict, q=fdr_level)
    
    lv1_center_boots = {
        k: hier_center_boots(v, n_bootstraps=n_bootstraps, stat_fun=stat_fun, rg=rg)
        for k, v in lv1_lv2_vals.items()}

    summary = {
        'conds': conds,
        'two-level_vals': lv1_lv2_vals,
        'two-level_nunit': lv1_lv2_nunit,
        'one-level_center_boots': lv1_center_boots,
        'fdr_level': fdr_level,
        # the below are None unless stat_test is given
        'stat_tests': stat_tests,
        'pvals_dict': pvals_dict,
        'apvals_dict': apvals_dict,
        'reject_dict': reject_dict}
    
    return summary
