#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 7c,d
# (- mas = model-perf-map along saccade)
script=7a_mas_match_control
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb  "$OUT_DIR"/log_${script}-${SESS}.ipynb