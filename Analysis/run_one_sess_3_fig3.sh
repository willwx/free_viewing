#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 3b, c
script=2d_rsc_1pt
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# Fig. 3f-h
script=2a_self_consistency
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# Fig. 3g dashed line ('default', i.e., no decorr)
# also provides pre-computed return fixations for several later scripts
OUT_DIR2="$OUT_DIR"/self_consistency_no_decorr
mkdir -p "$OUT_DIR2"
papermill -p sess_name ${SESS} -p min_sep 0 -p n_boots 0 -p n_perm 0 -p output_dir "$OUT_DIR2" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb