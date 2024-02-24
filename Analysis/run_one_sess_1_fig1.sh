#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 1g
script=3b_trial_level_psth
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb