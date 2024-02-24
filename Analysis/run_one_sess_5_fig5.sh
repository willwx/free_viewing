#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 5a-c; see run_one_sess_hier_group.sh

# Fig. 5e,f
script=1b_face_specific_match_control
papermill -p sess_name ${SESS} -p fsn_dir "$OUT_DIR" -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# Fig. 5h,i
script=2b_self_consistency_match_control
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb