#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'

# Fig. 2b,c,e
script=1a_face_specific
papermill -p sess_name ${SESS} -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb