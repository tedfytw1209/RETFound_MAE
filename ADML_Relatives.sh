#! /bin/bash

## study2 (original) launcher for ad_control / mci_control / ad_mci_control
## (see ADML_Relatives2.sh for study3)
##
## Usage:
##   bash ADML_Relatives.sh [TASK] [CW_MODE] [CLEAN] [START_FOLD]
##     TASK       : ad_control | mci_control | ad_mci_control  (default: ad_control)
##     CW_MODE    : none | cw    (add class weighting)         (default: none)
##     CLEAN      : clean | raw  (disease-clean vs original)   (default: clean)
##     START_FOLD : resume CV from this fold                   (default: 0)
##
## Examples:
##   bash ADML_Relatives.sh ad_control              # study2 disease-clean, ad_control
##   bash ADML_Relatives.sh ad_control cw           # + class weighting
##   bash ADML_Relatives.sh ad_control none raw     # original (non-disease-clean) study2
##   bash ADML_Relatives.sh ad_mci_control cw       # 3-class, class-weighted

TASK=${1:-ad_control}         # ad_control | mci_control | ad_mci_control
CW_MODE=${2:-none}            # none | cw
CLEAN=${3:-clean}            # clean | raw
START_FOLD=${4:-0}
if [ "$CW_MODE" = "cw" ]; then CW="--class_weight"; else CW=""; fi

# ------------------------------------------------------------------
# Derived: number of classes + split subdir per task
#   ad_control     -> 2 classes, split
#   mci_control    -> 2 classes, split_mcicon
#   ad_mci_control -> 3 classes, split_admcicon
# ------------------------------------------------------------------
case "$TASK" in
    ad_control)     NUM_CLASS=2; SUBDIR=split ;;
    mci_control)    NUM_CLASS=2; SUBDIR=split_mcicon ;;
    ad_mci_control) NUM_CLASS=3; SUBDIR=split_admcicon ;;
    *) echo "Unknown TASK: $TASK (use ad_control | mci_control | ad_mci_control)"; exit 1 ;;
esac
STUDY=${TASK}_detect_data     # CSV base name, e.g. ad_control_detect_data.csv

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

# ------------------------------------------------------------------
# Submit one job per relative model (uncomment the ones you want to run)
# ------------------------------------------------------------------

#Jacq (convnext_tiny, thickness)
sbatch finetune_Jacqueline_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR convnext_tiny 128 1e-3 5e-4 $NUM_CLASS 0 0.001 0.2 $START_FOLD $CW

#Wisely (resnet18_paper, thickness)
sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR resnet18_paper 0.01 1e-3 0.01 $NUM_CLASS 0 $START_FOLD $CW

#Mahendran (ad_oct_model, OCT)
sbatch finetune_Mahendran_ad_oct_model_cv.sh $STUDY $DATA $SPLIT_DIR ad_oct_model 256 3 false 7e-5 1e-2 $NUM_CLASS $START_FOLD $CW

#hebei (ducan, dual)
#sbatch finetune_Hebei_admci_detection_cv.sh $STUDY $DATA $SPLIT_DIR ducan 0.7 0.7 1.0 3e-4 1e-2 $NUM_CLASS $START_FOLD $CW

#Wisely2 (dual_input_cnn, images_only)
#sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR dual_input_cnn images_only 0.01 1e-4 0.01 $NUM_CLASS 0 10 $START_FOLD $CW

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
