#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 4a
script=2e_rsc_precision
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# Fig. 4c
script=3a_trial_level_sc
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb