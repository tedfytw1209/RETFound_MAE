#! /bin/bash
# Retrain the ad_control (AD vs control) models: RETFound and SMP-fuse, on IRB2024v5_ADCON_DL_data.
# Based on baseline_multirun_irb2024_ad.sh / baseline_multirun_irb2024_ad_tmp.sh.
#
#   bash baseline_multirun_irb2024_ad_control.sh            # RETFound + SMP-fuse
#   bash baseline_multirun_irb2024_ad_control.sh retfound   # one model only (retfound | smp)
#   DRY_RUN=1 bash baseline_multirun_irb2024_ad_control.sh  # print the sbatch commands, submit nothing
# env: DATASETS="..."   default: ad_control_detect_data (also: ad_mci_detect_data mci_control_detect_data)
#      ADDCMD="..."     extra training flag for both models (default: --bal_sampler, as in baseline_multirun_irb2024_ad.sh)
#      FORCE=1          submit even if the run already has a checkpoint-best.pth (it would be overwritten)
#      DATA_DIR=...     folder holding <dataset>.csv
#                       (default: /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data_retinaf_diseaseclean/, study2 disease-clean;
#                        others: IRB2024v5_ADCON_DL_data_retinaf, IRB2024v5_ADCON_DL_study3_retinaf_1yr / _10yr)
#
# main_finetune.py / main_finetune_smp.py take train/val/test from the CSV's `split` column; a CSV without it would
# be used whole for all three splits. The CSV is therefore checked (exists, has `split`) before anything is submitted.
#
# Note: baseline_multirun_irb2024_ad.sh passes Epochs as the 5th argument, but finetune_retfound_adcon_irb2024_v5.sh
# reads the 5th argument as Num_CLASS and the 6th as weight_decay (Epochs is fixed at 50 inside it). This script
# uses the order the training scripts expect.

WHICH=${1:-all} # all | retfound | smp
case "$WHICH" in all|retfound|smp) ;; *) echo "unknown model '$WHICH' (all|retfound|smp)" >&2; exit 1 ;; esac
DATASETS=(${DATASETS:-ad_control_detect_data})
ADDCMD=${ADDCMD-"--bal_sampler"}
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
export DATA_DIR=${DATA_DIR:-/blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data_retinaf_diseaseclean/} # read by both training scripts (sbatch exports env)
data_type=$(basename "$DATA_DIR") # the training scripts put this folder name in the run name

# RETFound (finetune_retfound_adcon_irb2024_v5.sh: 50 epochs, bs16, input 224)
RF_MODEL=RETFound_mae; RF_FT=RETFound_mae_natureOCT; RF_LR=5e-4; RF_WD=0.05; RF_EVAL=default

# SMP-fuse: the UF SMP configuration of multiseed_run_all.sh (100 epochs, bs16, input 512)
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
SMP_LR=5e-4; SMP_WD=1e-4; SMP_EVAL=default
SMP_MODE="fuse"; SMP_FUSE="weighted_sum"; SMP_ALPHA="0.5"; SMP_SIZE_MATCH="decoder_to_encoder"
SMP_FUSION_DIM="9"; SMP_ALIGN="pre"; SMP_ENC_IDX="-2"; SMP_DEC_IDX="-1"; SMP_CLASSIFIER="conv"
SMP_ALPHA_TYPE="scalar"; SMP_ADD1="--seg_mask"; SMP_ADD2="--smp_learnable_alpha"

num_class() { case "$1" in ad_mci_control_detect_data) echo 3 ;; *) echo 2 ;; esac; }

# run dirs = the training scripts' --task, so the checkpoint can be found (and checked) before submitting
rf_run()  { echo "$1-${data_type}-all-${RF_MODEL}-OCT-${RF_EVAL}eval-trsub0-${ADDCMD}-"; }
smp_run() { echo "$1-${data_type}-all-${SEG_CKPT}-OCT-bs16ep100lr${SMP_LR}optadamw-${SMP_EVAL}eval-trsub0-${SMP_MODE}-smp${SMP_FUSE}-${SMP_ALIGN}-${SMP_FUSION_DIM}-fea${SMP_ENC_IDX}${SMP_DEC_IDX}-${SMP_ALPHA}-${SMP_SIZE_MATCH}-${SMP_CLASSIFIER}-alpha${SMP_ALPHA_TYPE}-${SMP_ADD1}-${SMP_ADD2}-${ADDCMD}"; }

check_csv() { # <dataset>: the CSV must exist and carry the train/val/test `split` column
    local csv="${DATA_DIR%/}/$1.csv" err=""
    if [ ! -f "$csv" ]; then
        err="not found: $csv"
    elif ! head -1 "$csv" | tr -d '\r"' | tr ',' '\n' | grep -qx split; then
        err="$csv has no 'split' column; main_finetune*.py would train and test on the same rows"
    fi
    [ -z "$err" ] && return 0
    echo "ERROR: $err" >&2
    echo "  ADCON data folders: $(ls -d "$(dirname "${DATA_DIR%/}")"/IRB2024v5_ADCON_DL_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" >&2
    echo "  choose one with DATA_DIR=/path/to/folder/ bash $0 ..." >&2
    [ "${DRY_RUN:-0}" = "1" ]   # a dry run reports the problem and still prints the commands
}

submit() { # <run dir> <sbatch args...>
    local ckpt="$RESULTS_DIR/$1/checkpoint-best.pth"
    shift
    if [ -f "$ckpt" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] checkpoint exists: $ckpt (FORCE=1 to retrain)"
        return
    fi
    echo "sbatch $*"
    echo "  -> $ckpt"
    [ "${DRY_RUN:-0}" = "1" ] || sbatch "$@"
}

for DATASET in "${DATASETS[@]}"; do
    NUM_CLASS=$(num_class "$DATASET")
    echo "Running dataset: $DATASET ($DATA_DIR) with num_class=$NUM_CLASS"
    check_csv "$DATASET" || exit 1
    if [ "$WHICH" != "smp" ]; then
        submit "$(rf_run $DATASET)" finetune_retfound_adcon_irb2024_v5.sh $DATASET $RF_MODEL $RF_FT $RF_LR $NUM_CLASS $RF_WD \
            $RF_EVAL OCT 0 $ADDCMD
    fi
    if [ "$WHICH" != "retfound" ]; then
        submit "$(smp_run $DATASET)" finetune_retfound_adcon_irb2024_v5_smp.sh $DATASET SMP $SEG_CKPT $SMP_LR $NUM_CLASS $SMP_WD \
            $SMP_EVAL OCT 0 $SMP_MODE $SMP_FUSE $SMP_ALPHA $SMP_SIZE_MATCH $SMP_FUSION_DIM $SMP_ALIGN $SMP_ENC_IDX $SMP_DEC_IDX \
            $SMP_CLASSIFIER $SMP_ALPHA_TYPE $SMP_ADD1 $SMP_ADD2 $ADDCMD
    fi
done
