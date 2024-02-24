#!/bin/bash


# For rationale, see Methods (Per-neuron parameter estimates (latency and RF)).
# Results from here underlie Fig. 5a-c and derive, via scripts in Postprocessing/,
# some data tables in db/ (specified below); those tables are in turn used in
# some analyses (also specified below).


SESS=Pa210201
OUT_DIR='../test_results/hier_group'
mkdir -p "$OUT_DIR"


# A. Per-unit latencies
#    Derives: db/per_(hier_)unit_latency-(fix_|stim_)on.csv.gz
#    Used in:
#    - Fig. 5a-c
#    - These scripts: 1a 2d 2e 5a 6a 6b (only fix-on latencies)

# 1. Hierarchically group (sum) responses (i.e., spike-density functions or SDFs)
#    (note this script outputs to the data folder)
script=u0_hier_group_data
# (the highly quoted flags are just Python str reprs of lists; they will be parsed by ast in the script)
flags=( -p unit_dset sdf.unit_names -p dsets "['sdf']" -p unit_axes "[-1]" -p dsets_to_copy "['config/sdf']" )
sdf='-sdf-mwa_50'
papermill -p sess_name ${SESS} "${flags[@]}" -p input_suffix $sdf -p output_suffix "$sdf"-hg script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# 2. Calculate fixation onset-aligned latencies
sfx=-mwa_50-hg
script=2a_self_consistency
papermill -p sess_name ${SESS} -p sdf_suffix $sfx -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb
script=2u1_crossing_point
papermill -p sess_name ${SESS} -p rsc_dir "$OUT_DIR" -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# 3. Calculate stimulus onset-aligned latencies
script=2c_fix0_self_consistency
papermill -p sess_name ${SESS} -p sdf_suffix $sfx -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb
script=2u2_t2hh
papermill -p sess_name ${SESS} -p rsc_dir "$OUT_DIR" -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb


# B. Per-unit RFs
#    Derives: db/per_unit_rf.csv.gz
#    Used in: 1a 5a 7a

# 1. Do hier-group for sdf-mwa_1, used for fixation-aligned RFs (see *fig6.sh)
script=u0_hier_group_data
sdf='-sdf-mwa_1'
papermill -p sess_name ${SESS} "${flags[@]}" -p input_suffix $sdf -p output_suffix "$sdf"-hg  script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# 2. Compute model-based RF maps (see *fig6.sh)
sfx=-mwa_1-hg
OUT_DIR2="$OUT_DIR"/maps_fix_hg
mkdir -p "$OUT_DIR2"
flags=( -p sdf_suffix "$sfx" -p latency_path '../db/per_hier_unit_latency-fix_on.csv.gz' -p t_aln fix_on -p t_win 200 -p ifix_sel 2 -p select_saccades False -p sdf_suffix -mwa_1-hg -p output_dir "$OUT_DIR2" )
script=6a_model_perf_map
papermill -p sess_name ${SESS} "${flags[@]}" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb
script=6b_feat_corr_map
papermill -p sess_name ${SESS} "${flags[@]}" -p rfmap_dir "$OUT_DIR2" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb

# 3. Fit Gaussian to RFs
script=6u1_rf_gaussian_fit
papermill -p sess_name ${SESS} -p rf_map_analysis feat_corr_map -p rfmap_dir "$OUT_DIR2" -p output_dir "$OUT_DIR2" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb