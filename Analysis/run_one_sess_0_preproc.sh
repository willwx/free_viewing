#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

script=0a_shared_processing
papermill -p sess_name ${SESS} script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

script=0b_spike_density_function
papermill -p sess_name ${SESS} script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

script=0c_select_fix_sacc
papermill -p sess_name ${SESS} script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb
