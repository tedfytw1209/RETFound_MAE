#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --time=01:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# KDD_alpha_report.sh
#
# Reports the trained alpha (fusion gate) value for SMP-fuse-weighted_sum
# across all tasks:
#   IRB2024 tasks (own trained checkpoint per dataset):
#     DME_binary_all_split, AMD_all_split,
#     Glaucoma_binary_all_split, ERM_all_split
#   Public dataset tasks (each trained directly on the public dataset):
#     OCTDL_DME (study: DME_all, lr=1e-4, ep=50)
#     OCTDL_AMD (study: AMD_all, lr=1e-4, ep=50)
#     OCTDL_ERM (study: ERM_all, lr=1e-4, ep=50)
#     CellData  (study: DME_all, lr=1e-4, ep=50)
#
# Architecture is fixed for all experiments:
#   mode=fuse, fuse_mode=weighted_sum, enc_idx=-2, dec_idx=-1, fusion_dim=9
#
# No FLOPs or inference timing — CPU-only, loads checkpoint and reads alpha.
#
# Output CSV per task: ${OUTPUT_DIR}/KDD_alpha_<TASK>.csv
#
# Usage (interactive):  bash KDD_alpha_report.sh
# Usage (SLURM):        sbatch KDD_alpha_report.sh
# ─────────────────────────────────────────────────────────────────────────────

date; hostname; pwd

module load conda
conda activate octxai

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_TYPE=IRB2024_v5_all
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

# Fuse-weighted_sum checkpoint suffixes (lr and epochs differ between datasets)
IRB2024_FUSE_WS_SUFFIX="OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
PUBLIC_FUSE_WS_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run alpha extraction for one task
# Args: $1=TASK_LABEL  $2=RESUME_PATH
# ─────────────────────────────────────────────────────────────────────────────
run_task() {
    local TASK_LABEL=$1
    local RESUME=$2

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Task: ${TASK_LABEL}"
    echo "  Checkpoint: ${RESUME}"
    echo "════════════════════════════════════════════════════════════════════"

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
        --smp_resumes       "" "" "${RESUME}" "" \
        --smp_modes         fuse \
        --smp_fuse_modes    weighted_sum \
        --skip_baselines \
        --alpha_only \
        --device            cpu \
        --output_csv        "${OUTPUT_DIR}/KDD_alpha_${TASK_LABEL}.csv"
}

# ── IRB2024 tasks — each dataset has its own trained checkpoint ───────────────
for DATASET in DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split; do
    RESUME=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-${IRB2024_FUSE_WS_SUFFIX}/checkpoint-best.pth
    run_task "${DATASET}" "${RESUME}"
done

# ── Public dataset tasks — each has its own trained checkpoint ────────────────
# Checkpoint pattern: {STUDY}-{OCTDL|CellData}-all-${SEG_CKPT}-OCT-bs16ep50lr1e-4...
# (trained directly on public datasets, not transferred from IRB2024)
OCTDL_DME_RESUME=${RESULTS_DIR}/DME_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth
OCTDL_AMD_RESUME=${RESULTS_DIR}/AMD_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth
OCTDL_ERM_RESUME=${RESULTS_DIR}/ERM_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth
CELLDATA_RESUME=${RESULTS_DIR}/DME_all-CellData-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth

run_task "OCTDL_DME" "${OCTDL_DME_RESUME}"
run_task "OCTDL_AMD" "${OCTDL_AMD_RESUME}"
run_task "OCTDL_ERM" "${OCTDL_ERM_RESUME}"
run_task "CellData"  "${CELLDATA_RESUME}"

echo ""
echo "All tasks complete. CSVs written to ${OUTPUT_DIR}/KDD_alpha_*.csv"
