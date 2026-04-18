#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# KDD_ECE_baseline_report.sh
#
# Computes ECE for all baseline models across all 8 KDD tasks.
# XAI evaluation is skipped (--skip_xai).
#
# Baseline models:
#   RETFound_mae       (finetune: RETFound_mae_natureOCT,          input: 224)
#   timm_efficientnet  (finetune: timm_efficientnet-b4,            input: 380 IRB2024 / 224 public)
#   ViT-B/16           (finetune: google/vit-base-patch16-224-in21k, input: 224)
#   ResNet-50          (finetune: microsoft/resnet-50,             input: 224)
#
#   IRB2024 tasks (ep50, lr5e-4, trsub0):
#     DME_binary_all_split, AMD_all_split,
#     Glaucoma_binary_all_split, ERM_all_split
#   Public dataset tasks:
#     OCTDL_DME (study: DME_all, ep50, lr5e-4)
#     OCTDL_AMD (study: AMD_all, ep50, lr5e-4)
#     OCTDL_ERM (study: ERM_all, ep50, lr5e-4)
#     CellData  (study: DME_all, ep3,  lr5e-4)
#
# Output: ${OUTPUT_DIR}/output_dir/<TASK_LABEL>-ECE/ece_test.csv
#
# Usage (interactive):  bash KDD_ECE_baseline_report.sh
# Usage (SLURM):        sbatch KDD_ECE_baseline_report.sh
# ─────────────────────────────────────────────────────────────────────────────

date; hostname; pwd

module load conda
conda activate octxai

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_TYPE=IRB2024_v5_all
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/RETfound_ECE_results

IRB2024_DATA_DIR=/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune8-eval2
IRB2024_IMG_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/
IRB2024_MASK_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/

OCTDL_DATA_DIR=/orange/ruogu.fang/tienyuchang/OCTDL
OCTDL_IMG_DIR=/orange/ruogu.fang/tienyuchang/OCTDL/
OCTDL_MASK_DIR=/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/

CELLDATA_DATA_DIR=/orange/ruogu.fang/tienyuchang/CellData/OCT
CELLDATA_IMG_DIR=/orange/ruogu.fang/tienyuchang/CellData/
CELLDATA_MASK_DIR=/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/

cd /blue/ruogu.fang/tienyuchang/RETFound_MAE || exit 1

# ── Common settings ───────────────────────────────────────────────────────────
NB_CLASSES=2

# ── Checkpoint suffixes ───────────────────────────────────────────────────────
IRB2024_BASELINE_SUFFIX="OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--"
OCTDL_BASELINE_SUFFIX="OCT-bs16ep50lr5e-4optadamw-defaulteval--"
CELLDATA_BASELINE_SUFFIX="OCT-bs16ep3lr5e-4optadamw-defaulteval--"

# ── Baseline model selection (comment out models to skip) ─────────────────────
# Input size: timm_efficientnet-b4 uses 380 for IRB2024, 224 for public datasets
MODEL_NAMES=(); MODEL_FINETUNES=(); MODEL_LABELS=()

MODEL_NAMES+=("RETFound_mae");        MODEL_FINETUNES+=("RETFound_mae_natureOCT");               MODEL_LABELS+=("RETFound")
MODEL_NAMES+=("timm_efficientnet-b4"); MODEL_FINETUNES+=("timm_efficientnet-b4");                MODEL_LABELS+=("EfficientNet-b4")
MODEL_NAMES+=("vit-base-patch16-224"); MODEL_FINETUNES+=("google/vit-base-patch16-224-in21k");   MODEL_LABELS+=("ViT-B16")
MODEL_NAMES+=("resnet-50");           MODEL_FINETUNES+=("microsoft/resnet-50");                  MODEL_LABELS+=("ResNet-50")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run ECE for one task
# Args: $1=TASK_LABEL  $2=MODEL  $3=FINETUNE  $4=RESUME
#       $5=INPUT_SIZE  $6=DATA_CSV  $7=IMG_DIR  $8=MASK_DIR
# ─────────────────────────────────────────────────────────────────────────────
run_task() {
    local TASK_LABEL=$1
    local MODEL=$2
    local FINETUNE=$3
    local RESUME=$4
    local INPUT_SIZE=$5
    local DATA_CSV=$6
    local IMG_DIR=$7
    local MASK_DIR=$8

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Task:       ${TASK_LABEL}"
    echo "  Model:      ${MODEL} / ${FINETUNE}"
    echo "  Input size: ${INPUT_SIZE}"
    echo "  Checkpoint: ${RESUME}"
    echo "  Data:       ${DATA_CSV}"
    echo "════════════════════════════════════════════════════════════════════"

    TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py \
        --model          "${MODEL}" \
        --finetune       "${FINETUNE}" \
        --nb_classes     ${NB_CLASSES} \
        --input_size     ${INPUT_SIZE} \
        --data_path      "${DATA_CSV}" \
        --img_dir        "${IMG_DIR}" \
        --thickness_dir  "${MASK_DIR}" \
        --task           "${TASK_LABEL}-ECE" \
        --output_dir     "${OUTPUT_DIR}/output_dir" \
        --log_dir        "${OUTPUT_DIR}/output_logs" \
        --resume         "${RESUME}" \
        --num_workers    8 \
        --batch_size     8 \
        --skip_xai
}

# ── Loop over all baseline models ─────────────────────────────────────────────
for i in "${!MODEL_NAMES[@]}"; do
    MODEL="${MODEL_NAMES[$i]}"
    FINETUNE="${MODEL_FINETUNES[$i]}"
    LABEL="${MODEL_LABELS[$i]}"

    # timm_efficientnet-b4 was trained at 380 for IRB2024, 224 for public datasets
    if [[ "${MODEL}" == "timm_efficientnet-b4" ]]; then
        IRB2024_INPUT_SIZE=380
    else
        IRB2024_INPUT_SIZE=224
    fi
    PUBLIC_INPUT_SIZE=224

    # ── IRB2024 tasks (ep50, lr5e-4, trsub0) ──────────────────────────────────
    for DATASET in DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split; do
        RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${FINETUNE}-${IRB2024_BASELINE_SUFFIX}/checkpoint-best.pth"
        run_task "${DATASET}-${LABEL}" "${MODEL}" "${FINETUNE}" "${RESUME}" "${IRB2024_INPUT_SIZE}" \
            "${IRB2024_DATA_DIR}/${DATASET}.csv" \
            "${IRB2024_IMG_DIR}" "${IRB2024_MASK_DIR}"
    done

    # ── OCTDL tasks (ep50, lr5e-4) ────────────────────────────────────────────
    OCTDL_DME_RESUME="${RESULTS_DIR}/DME_all-OCTDL-all-${FINETUNE}-${OCTDL_BASELINE_SUFFIX}/checkpoint-best.pth"
    OCTDL_AMD_RESUME="${RESULTS_DIR}/AMD_all-OCTDL-all-${FINETUNE}-${OCTDL_BASELINE_SUFFIX}/checkpoint-best.pth"
    OCTDL_ERM_RESUME="${RESULTS_DIR}/ERM_all-OCTDL-all-${FINETUNE}-${OCTDL_BASELINE_SUFFIX}/checkpoint-best.pth"

    run_task "OCTDL_DME-${LABEL}" "${MODEL}" "${FINETUNE}" "${OCTDL_DME_RESUME}" "${PUBLIC_INPUT_SIZE}" \
        "${OCTDL_DATA_DIR}/DME_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

    run_task "OCTDL_AMD-${LABEL}" "${MODEL}" "${FINETUNE}" "${OCTDL_AMD_RESUME}" "${PUBLIC_INPUT_SIZE}" \
        "${OCTDL_DATA_DIR}/AMD_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

    run_task "OCTDL_ERM-${LABEL}" "${MODEL}" "${FINETUNE}" "${OCTDL_ERM_RESUME}" "${PUBLIC_INPUT_SIZE}" \
        "${OCTDL_DATA_DIR}/ERM_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

    # ── CellData task (ep3, lr5e-4) ───────────────────────────────────────────
    CELLDATA_RESUME="${RESULTS_DIR}/DME_all-CellData-all-${FINETUNE}-${CELLDATA_BASELINE_SUFFIX}/checkpoint-best.pth"
    run_task "CellData-${LABEL}" "${MODEL}" "${FINETUNE}" "${CELLDATA_RESUME}" "${PUBLIC_INPUT_SIZE}" \
        "${CELLDATA_DATA_DIR}/DME_all.csv" "${CELLDATA_IMG_DIR}" "${CELLDATA_MASK_DIR}"

done

echo ""
echo "All tasks complete. Results written to ${OUTPUT_DIR}/output_dir/*/ece_test.csv"
