#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# KDD_computation_eval.sh
#
# Measures parameter / FLOPs / inference-time overhead for SMP models
# (enc / dec / fuse-weighted_sum / fuse-multiply) across all 8 tasks:
#
#   IRB2024 (UF) tasks  — ep100, lr5e-4:
#     UF_DME       DME_binary_all_split   IRB2024_v5_all
#     UF_AMD       AMD_all_split          IRB2024_v5_all
#     UF_Glaucoma  Glaucoma_binary_all_split  IRB2024_v5_all
#     UF_ERM       ERM_all_split          IRB2024_v5_all
#
#   Public tasks — ep50, lr1e-4:
#     OCTDL_DME    DME_all                OCTDL
#     OCTDL_AMD    AMD_all                OCTDL
#     OCTDL_ERM    ERM_all                OCTDL
#     CellData_DME DME_all                CellData
#
# One CSV per task: ${OUTPUT_DIR}/KDD_computation_<TASK>.csv
#
# Usage (interactive):  bash KDD_computation_eval.sh
# Usage (SLURM):        sbatch KDD_computation_eval.sh
# ─────────────────────────────────────────────────────────────────────────────

date; hostname; pwd

module load conda
conda activate octxai

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth

cd /blue/ruogu.fang/tienyuchang/RETFound_MAE || exit 1

# ── SMP architecture (fixed for all KDD experiments) ─────────────────────────
SEG_ARCH=Unet
ENCODER_NAME=resnet50
IN_CHANNELS=3
NB_CLASSES=2
SEG_CLASSES=9
SEG_ACTIVATION=softmax
INPUT_SIZE=512
ALPHA=0.5
SIZE_MATCH=decoder_to_encoder
SMP_CLASSIFIER=conv
ALIGN=pre

# ── Timing ────────────────────────────────────────────────────────────────────
N_WARMUP=10
N_RUNS=50
BATCH_SIZE=1
DEVICE=cuda

# ── Checkpoint suffix patterns ────────────────────────────────────────────────
# IRB2024 (UF): ep100, lr5e-4
IRB_ENC_SUFFIX="OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
IRB_DEC_SUFFIX="OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
IRB_FUSE_WS_SUFFIX="OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
IRB_FUSE_MUL_SUFFIX="OCT-bs16ep100lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--"

# Public datasets (OCTDL): ep50, lr1e-4
PUB_ENC_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
PUB_DEC_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
PUB_FUSE_WS_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
PUB_FUSE_MUL_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--"

# CellData: ep5, lr1e-4
CELL_ENC_SUFFIX="OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
CELL_DEC_SUFFIX="OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---"
CELL_FUSE_WS_SUFFIX="OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
CELL_FUSE_MUL_SUFFIX="OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--"

# ─────────────────────────────────────────────────────────────────────────────
# Set DRY_RUN=1 to print all resume paths and existence checks without running.
# ─────────────────────────────────────────────────────────────────────────────
DRY_RUN=${DRY_RUN:-0}

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run computation eval for one task
# Args: $1=TASK_LABEL  $2=DATASET  $3=DATA_TYPE
#       $4=ENC_SUFFIX  $5=DEC_SUFFIX  $6=FUSE_WS_SUFFIX  $7=FUSE_MUL_SUFFIX
# ─────────────────────────────────────────────────────────────────────────────
run_task() {
    local TASK_LABEL=$1
    local DATASET=$2
    local DATA_TYPE=$3
    local ENC_SUFFIX=$4
    local DEC_SUFFIX=$5
    local FUSE_WS_SUFFIX=$6
    local FUSE_MUL_SUFFIX=$7

    local BASE="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}"
    local RESUME_ENC="${BASE}-${ENC_SUFFIX}/checkpoint-best.pth"
    local RESUME_DEC="${BASE}-${DEC_SUFFIX}/checkpoint-best.pth"
    local RESUME_FUSE_WS="${BASE}-${FUSE_WS_SUFFIX}/checkpoint-best.pth"
    local RESUME_FUSE_MUL="${BASE}-${FUSE_MUL_SUFFIX}/checkpoint-best.pth"

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Task: ${TASK_LABEL}  (${DATASET} / ${DATA_TYPE})"
    echo "════════════════════════════════════════════════════════════════════"
    echo "  [enc]      $([ -f "${RESUME_ENC}"      ] && echo OK   || echo MISSING) ${RESUME_ENC}"
    echo "  [dec]      $([ -f "${RESUME_DEC}"      ] && echo OK   || echo MISSING) ${RESUME_DEC}"
    echo "  [fuse_ws]  $([ -f "${RESUME_FUSE_WS}"  ] && echo OK   || echo MISSING) ${RESUME_FUSE_WS}"
    echo "  [fuse_mul] $([ -f "${RESUME_FUSE_MUL}" ] && echo OK   || echo MISSING) ${RESUME_FUSE_MUL}"

    [ "${DRY_RUN}" = "1" ] && return

    python KDD_computation_eval.py \
        --seg_arch          ${SEG_ARCH} \
        --encoder_name      ${ENCODER_NAME} \
        --in_channels       ${IN_CHANNELS} \
        --nb_classes        ${NB_CLASSES} \
        --seg_classes       ${SEG_CLASSES} \
        --seg_activation    ${SEG_ACTIVATION} \
        --input_size        ${INPUT_SIZE} \
        --smp_fuse_mode     weighted_sum \
        --smp_learnable_alpha \
        --alpha             ${ALPHA} \
        --size_match        ${SIZE_MATCH} \
        --use_mask \
        --smp_classifier    ${SMP_CLASSIFIER} \
        --align             ${ALIGN} \
        --seg_ckpt          "${SEG_CKPT}" \
        --smp_resumes       "${RESUME_ENC}" "${RESUME_DEC}" \
                            "${RESUME_FUSE_WS}" "${RESUME_FUSE_MUL}" \
        --skip_baselines \
        --n_warmup          ${N_WARMUP} \
        --n_runs            ${N_RUNS} \
        --batch_size        ${BATCH_SIZE} \
        --device            ${DEVICE} \
        --output_csv        "${OUTPUT_DIR}/KDD_computation_${TASK_LABEL}.csv"
}

# ── IRB2024 (UF) tasks ────────────────────────────────────────────────────────
run_task "UF_DME"      "DME_binary_all_split"       "IRB2024_v5_all" \
    "${IRB_ENC_SUFFIX}" "${IRB_DEC_SUFFIX}" "${IRB_FUSE_WS_SUFFIX}" "${IRB_FUSE_MUL_SUFFIX}"

run_task "UF_AMD"      "AMD_all_split"              "IRB2024_v5_all" \
    "${IRB_ENC_SUFFIX}" "${IRB_DEC_SUFFIX}" "${IRB_FUSE_WS_SUFFIX}" "${IRB_FUSE_MUL_SUFFIX}"

run_task "UF_Glaucoma" "Glaucoma_binary_all_split"  "IRB2024_v5_all" \
    "${IRB_ENC_SUFFIX}" "${IRB_DEC_SUFFIX}" "${IRB_FUSE_WS_SUFFIX}" "${IRB_FUSE_MUL_SUFFIX}"

run_task "UF_ERM"      "ERM_all_split"              "IRB2024_v5_all" \
    "${IRB_ENC_SUFFIX}" "${IRB_DEC_SUFFIX}" "${IRB_FUSE_WS_SUFFIX}" "${IRB_FUSE_MUL_SUFFIX}"

# ── Public dataset tasks ──────────────────────────────────────────────────────
run_task "OCTDL_DME"    "DME_all" "OCTDL" \
    "${PUB_ENC_SUFFIX}" "${PUB_DEC_SUFFIX}" "${PUB_FUSE_WS_SUFFIX}" "${PUB_FUSE_MUL_SUFFIX}"

run_task "OCTDL_AMD"    "AMD_all" "OCTDL" \
    "${PUB_ENC_SUFFIX}" "${PUB_DEC_SUFFIX}" "${PUB_FUSE_WS_SUFFIX}" "${PUB_FUSE_MUL_SUFFIX}"

run_task "OCTDL_ERM"    "ERM_all" "OCTDL" \
    "${PUB_ENC_SUFFIX}" "${PUB_DEC_SUFFIX}" "${PUB_FUSE_WS_SUFFIX}" "${PUB_FUSE_MUL_SUFFIX}"

run_task "CellData_DME" "DME_all" "CellData" \
    "${CELL_ENC_SUFFIX}" "${CELL_DEC_SUFFIX}" "${CELL_FUSE_WS_SUFFIX}" "${CELL_FUSE_MUL_SUFFIX}"

echo ""
echo "All tasks complete. CSVs written to ${OUTPUT_DIR}/KDD_computation_*.csv"
