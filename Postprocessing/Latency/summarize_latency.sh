#!/bin/bash

script="Summarize latency"
res_fix="self_consistency_boot200-cp"
res_stim="fix0_self_consistency_boot200-t2hh"

papermill -p analysis_name fix_on -p results_subdir "${res_fix}" "${script}.ipynb" "Log - ${script} - Fix on.ipynb"
papermill -p analysis_name stim_on -p results_subdir "${res_stim}" "${script}.ipynb" "Log - ${script} - Stim on.ipynb"
papermill -p analysis_name fix_on -p results_subdir "${res_fix}" -p output_sfx "-more" -p boots_spread 50 -p min_boots_frac 0.25 "${script}.ipynb" "Log - ${script} - Fix on, more.ipynb"