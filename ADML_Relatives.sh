#! /bin/bash

## study2 (original) launcher for ad_control / mci_control / ad_mci_control
## (see ADML_Relatives2.sh for study3)
##
## Usage:
##   bash ADML_Relatives.sh [TASK] [CW_MODE] [CLEAN] [START_FOLD] [TAGS]
##     TASK       : ad_control | mci_control | ad_mci_control | ad_rest  (default: ad_control)
##                  ad_rest = binary AD-vs-REST: reuses ad_control_detect_data.csv but
##                  treats both mci and control rows as the negative class (see
##                  main_finetune_Chua_Jacqueline.py's ad_rest handling)
##     CW_MODE    : none | cw    (add class weighting)         (default: none)
##     CLEAN      : clean | raw  (disease-clean vs original)   (default: clean)
##     START_FOLD : resume CV from this fold                   (default: 0)
##     TAGS       : comma-separated manual wandb tags, merged in addition
##                  to the auto-derived study/task/period tags  (default: none)
##
## Examples:
##   bash ADML_Relatives.sh ad_control              # study2 disease-clean, ad_control
##   bash ADML_Relatives.sh ad_control cw           # + class weighting
##   bash ADML_Relatives.sh ad_control none raw     # original (non-disease-clean) study2
##   bash ADML_Relatives.sh ad_mci_control cw       # 3-class, class-weighted
##   bash ADML_Relatives.sh ad_control none clean 0 rerun,sanity-check  # + manual tags
##   bash ADML_Relatives.sh ad_rest                 # study2 disease-clean, AD-vs-rest (mci+control)

TASK=${1:-ad_control}         # ad_control | mci_control | ad_mci_control
CW_MODE=${2:-none}            # none | cw
CLEAN=${3:-clean}            # clean | raw
START_FOLD=${4:-0}
TAGS=${5:-}                   # manual comma-separated wandb tags (merged with auto tags)
if [ "$CW_MODE" = "cw" ]; then CW="--class_weight"; else CW=""; fi
export EXTRA_TAGS="$TAGS"     # inherited by sbatch jobs; merged into WANDB_TAGS downstream

# ------------------------------------------------------------------
# Derived: number of classes + split subdir per task
#   ad_control     -> 2 classes, split
#   mci_control    -> 2 classes, split_mcicon
#   ad_mci_control -> 3 classes, split_admcicon
#   ad_rest        -> 2 classes, split_adrest (needs mci patients present to relabel as negative)
# ------------------------------------------------------------------
case "$TASK" in
    ad_control)     NUM_CLASS=2; SUBDIR=split ;;
    mci_control)    NUM_CLASS=2; SUBDIR=split_mcicon ;;
    ad_mci_control) NUM_CLASS=3; SUBDIR=split_admcicon ;;
    ad_rest)        NUM_CLASS=2; SUBDIR=split_adrest ;;
    *) echo "Unknown TASK: $TASK (use ad_control | mci_control | ad_mci_control | ad_rest)"; exit 1 ;;
esac
STUDY=${TASK}_detect_data     # CSV base name, e.g. ad_control_detect_data.csv
                              # (ad_rest -> ad_rest_detect_data; main_finetune_Chua_Jacqueline.py
                              #  redirects this to ad_control_detect_data.csv at load time)

# ------------------------------------------------------------------
# Study2 split + data locations (disease-clean vs original/raw).
# SPLIT_DIR is passed already resolved to the task subdir; the CV scripts detect
# the split* leaf and use it as-is.
# ------------------------------------------------------------------
if [ "$CLEAN" = "raw" ]; then
    SPLIT_BASE=/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf
    DATA=${DATA:-IRB2024v5_ADCON_DL_data_retinaf}
else
    SPLIT_BASE=/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean
    DATA=${DATA:-IRB2024v5_ADCON_DL_data_retinaf_diseaseclean}
fi
SPLIT_DIR=${SPLIT_BASE}/${SUBDIR}

echo "STUDY2  TASK=$TASK  NUM_CLASS=$NUM_CLASS  STUDY=$STUDY  CLEAN=$CLEAN  CW='${CW}'"
echo "DATA=$DATA"
echo "SPLIT_DIR=$SPLIT_DIR"
echo "EXTRA_TAGS=$EXTRA_TAGS"

# ------------------------------------------------------------------
# Submit one job per relative model (uncomment the ones you want to run)
# ------------------------------------------------------------------

#Jacq (convnext_tiny, thickness)
echo "sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR convnext_tiny 64 1e-3 5e-4 $NUM_CLASS 0 0.001 0.2 $START_FOLD $CW"
#sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR convnext_tiny 64 1e-3 5e-4 $NUM_CLASS 0 0.001 0.2 $START_FOLD $CW

#Wisely (resnet18_paper, thickness)
echo "sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR resnet18_paper 0.01 1e-3 0.01 $NUM_CLASS 0 $START_FOLD $CW"
sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR resnet18_paper 0.01 1e-3 0.01 $NUM_CLASS 0 $START_FOLD $CW

#Mahendran (ad_oct_model, OCT)
echo "sbatch finetune_Mahendran_ad_oct_model_cv.sh $STUDY $DATA $SPLIT_DIR ad_oct_model 256 3 false 7e-5 1e-2 $NUM_CLASS $START_FOLD $CW"
sbatch finetune_Mahendran_ad_oct_model_cv.sh $STUDY $DATA $SPLIT_DIR ad_oct_model 256 3 false 7e-5 1e-2 $NUM_CLASS $START_FOLD $CW

#hebei (ducan, dual)
echo "sbatch finetune_Hebei_admci_detection_cv.sh $STUDY $DATA $SPLIT_DIR ducan 0.7 0.7 1.0 3e-4 1e-2 $NUM_CLASS $START_FOLD $CW"
sbatch finetune_Hebei_admci_detection_cv.sh $STUDY $DATA $SPLIT_DIR ducan 0.7 0.7 1.0 3e-4 1e-2 $NUM_CLASS $START_FOLD $CW

#Wisely2 (dual_input_cnn, images_only)
echo "sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR dual_input_cnn images_only 0.01 1e-4 0.01 $NUM_CLASS 0 10 $START_FOLD $CW"
sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR dual_input_cnn images_only 0.01 1e-4 0.01 $NUM_CLASS 0 10 $START_FOLD $CW

# ==================================================================
# Reference: original study2 invocations (verbatim)
# ==================================================================
# --- disease-clean, class-weighted (ad_control) ---
#sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf_diseaseclean /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split convnext_tiny 128 1e-3 5e-4 2 0 0.001 0.2 0 --class_weight
#sbatch finetune_Hebei_admci_detection_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf_diseaseclean /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split ducan 0.7 0.7 0 3e-4 1e-2 2 0 --class_weight
#sbatch finetune_Mahendran_ad_oct_model_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf_diseaseclean /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split ad_oct_model 256 3 false 7e-5 1e-2 2 0 --class_weight
#sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf_diseaseclean /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split resnet18_paper 0.01 1e-3 0.01 2 0 0 --class_weight
#sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf_diseaseclean /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf_diseaseclean/split dual_input_cnn images_only 0.01 1e-4 0.01 2 0 10 6 --class_weight
#
# --- original/raw, non-class-weighted (ad_control) ---
#sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split convnext_tiny 128 1e-3 5e-4 2 0 0.001 0.2 0
#sbatch finetune_Hebei_admci_detection_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ducan 0.7 0.7 0 3e-4 1e-2 2 0
#sbatch finetune_Mahendran_ad_oct_model_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split ad_oct_model 256 3 false 7e-5 1e-2 2 0
#sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split resnet18_paper 0.01 1e-3 0.01 2 0 0
#sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data_retinaf /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split dual_input_cnn images_only 0.01 1e-4 0.01 2 0 10 0
