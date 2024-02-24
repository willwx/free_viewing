#!/bin/bash

SESS=Pa210201
OUT_DIR='../test_results'


# Fig. 6b
# - prereq: MWA with t_win=1 (equiv. to no MWA, but has the expected data format)
script=0b_spike_density_function
papermill -p sess_name ${SESS} -p sdf_window 1 script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb

# - separately do zeroth and non-zeroth fixations
script=5a_vision_model
ifixs=( 0 1 )
for ifix in "${ifixs[@]}"
do
   OUT_DIR_="$OUT_DIR"/fix$ifix
   mkdir -p "$OUT_DIR_"
   papermill -p sess_name ${SESS} -p t_win 200 -p sdf_suffix -mwa_1 -p ifix_sel $ifix -p output_dir "$OUT_DIR_" script_${script}.ipynb "$OUT_DIR_"/log_${script}-${SESS}.ipynb
done


# Fig. 6d-f
# - maps are first done as performance maps (script_6a), then (using the peak
#   performance location to fit model coefs) as spike-triggered-average
#   (STA)-like maps (script 6b)

# Fig. 6d
# - a 200-ms window per fixation
# - includes both zeroth and non-zeroth fixations (coded as `ifix_sel=2`)
# - requires mwa_1
OUT_DIR2="$OUT_DIR"/maps_fix
mkdir -p "$OUT_DIR2"
flags=( -p t_aln fix_on -p t_win 200 -p ifix_sel 2 -p select_saccades False -p sdf_suffix -mwa_1 -p output_dir "$OUT_DIR2" )
script=6a_model_perf_map
papermill -p sess_name ${SESS} "${flags[@]}" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb
script=6b_feat_corr_map  # requires 6a through rfmap_dir
papermill -p sess_name ${SESS} "${flags[@]}" -p rfmap_dir "$OUT_DIR2" script_${script}.ipynb "$OUT_DIR2"/log_${script}-${SESS}.ipynb

# Fig. 6e
# - like Fig. 6d but for saccades and sliding windows aligned to saccade onsets
# - the unit_sel_path option saves time by only analyzing selective units
flags=( -p unit_sel_path ../db/unit_sel/visually_selective.csv.gz -p output_dir "$OUT_DIR" )
script=6a_model_perf_map
papermill -p sess_name ${SESS} "${flags[@]}" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb
script=6b_feat_corr_map  # requires 6a through rfmap_dir
papermill -p sess_name ${SESS} "${flags[@]}" -p rfmap_dir "$OUT_DIR" script_${script}.ipynb  "$OUT_DIR"/log_${script}-${SESS}.ipynb

# Fig. 6f
# - 1D quantification of 3D maps in Fig. 6e
script=6u1_rf_gaussian_fit
papermill -p sess_name ${SESS} -p rfmap_analysis feat_corr_map -p rfmap_dir "$OUT_DIR" -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb
script=6u2_rf_cons_via_gauss_fit
papermill -p sess_name ${SESS} -p rfmap_analysis feat_corr_map -p rfmap_dir "$OUT_DIR" -p rffit_dir "$OUT_DIR" -p output_dir "$OUT_DIR" script_${script}.ipynb "$OUT_DIR"/log_${script}-${SESS}.ipynb