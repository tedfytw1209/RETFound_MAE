#! /bin/bash

## study3 (disease-clean) launcher for ad_control / mci_control / ad_mci_control
## (see ADML_Relatives.sh for study2 / original)
##
## Usage:
##   bash ADML_Relatives2.sh [TASK] [DATA] [PERIOD] [CW_MODE] [START_FOLD]
##     TASK       : ad_control | mci_control | ad_mci_control  (default: ad_mci_control)
##     DATA       : path to study3 data folder                 (default: IRB2024v5_ADCON_DL_study3_retinaf_1yr)
##     PERIOD     : split period, e.g. 1yr                     (default: 1yr)
##     CW_MODE    : none | cw    (add class weighting)         (default: none)
##     START_FOLD : resume CV from this fold                   (default: 0)
##
## Examples:
##   bash ADML_Relatives2.sh ad_control /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_study3_retinaf_1yr 1yr
##   bash ADML_Relatives2.sh mci_control /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_study3_retinaf_1yr 1yr
##   bash ADML_Relatives2.sh ad_mci_control /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_study3_retinaf_1yr 1yr cw    # 3-class, class-weighted

TASK=${1:-ad_mci_control}     # ad_control | mci_control | ad_mci_control
DATA=${2:-IRB2024v5_ADCON_DL_study3_retinaf_1yr}  # default study3 data folder
PERIOD=${3:-1yr}
CW_MODE=${4:-none}            # none | cw
START_FOLD=${5:-0}
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
STUDY=${TASK}_detect_data     # CSV base name, e.g. ad_mci_control_detect_data.csv

# ------------------------------------------------------------------
# Study3 (disease-clean) split + data locations.
# SPLIT_DIR is passed already resolved to the task subdir; the CV scripts detect
# the split* leaf and use it as-is.
# ------------------------------------------------------------------
SPLIT_BASE=/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_combined_study3_retinaf_diseaseclean_${PERIOD}
SPLIT_DIR=${SPLIT_BASE}/${SUBDIR}

# Folder under /blue/ruogu.fang/tienyuchang/ that holds ${STUDY}.csv for study3.
# All relatives read from the same folder; modality selects thickness vs paired images.
# Override via env if needed, e.g. DATA=... bash ADML_Relatives2.sh ...


echo "STUDY3  TASK=$TASK  NUM_CLASS=$NUM_CLASS  STUDY=$STUDY  PERIOD=$PERIOD  CW='${CW}'"
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
sbatch finetune_Hebei_admci_detection_cv.sh $STUDY $DATA $SPLIT_DIR ducan 0.7 0.7 1.0 3e-4 1e-2 $NUM_CLASS $START_FOLD $CW

#Wisely2 (dual_input_cnn, images_only)
sbatch finetune_Wisely2_adcon_irb2024_v5_cv.sh $STUDY $DATA $SPLIT_DIR dual_input_cnn images_only 0.01 1e-4 0.01 $NUM_CLASS 0 10 $START_FOLD $CW
