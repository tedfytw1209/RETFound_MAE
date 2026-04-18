#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# KDD_ECE_report.sh
#
# Computes Expected Calibration Error (ECE) for SMP-fuse-weighted_sum
# across all tasks.  XAI evaluation is skipped (--skip_xai).
#
#   IRB2024 tasks (own trained checkpoint per dataset):
#     DME_binary_all_split, AMD_all_split,
#     Glaucoma_binary_all_split, ERM_all_split
#   Public dataset tasks (each trained directly on the public dataset):
#     OCTDL_DME (study: DME_all, lr=1e-4, ep=50)
#     OCTDL_AMD (study: AMD_all, lr=1e-4, ep=50)
#     OCTDL_ERM (study: ERM_all, lr=1e-4, ep=50)
#     CellData  (study: DME_all, lr=1e-4, ep=5)
#
# Architecture fixed for all experiments:
#   mode=fuse, fuse_mode=weighted_sum, enc_idx=-2, dec_idx=-1, fusion_dim=9
#
# Output per task: ${OUTPUT_DIR}/output_dir/<TASK_LABEL>-ECE/ece_test.csv
#
# Usage (interactive):  bash KDD_ECE_report.sh
# Usage (SLURM):        sbatch KDD_ECE_report.sh
# ─────────────────────────────────────────────────────────────────────────────

date; hostname; pwd

module load conda
conda activate octxai

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_TYPE=IRB2024_v5_all
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
OUTPUT_DIR=/orange/ruogu.fang/tienyuchang/RETfound_ECE_results

SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth

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

# ── SMP architecture (fixed for all KDD experiments) ─────────────────────────
NB_CLASSES=2
INPUT_SIZE=512
ENC_IDX=-2
DEC_IDX=-1
FUSION_DIM=9
ALPHA=0.5
SIZE_MATCH=decoder_to_encoder
SMP_CLASSIFIER=conv
ALIGN=pre

# Fuse-weighted_sum checkpoint suffixes (lr and epochs differ between datasets)
IRB2024_FUSE_WS_SUFFIX="OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
PUBLIC_FUSE_WS_SUFFIX="OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"
CELLDATA_FUSE_WS_SUFFIX="OCT-bs16ep5lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run ECE for one task
# Args: $1=TASK_LABEL  $2=RESUME  $3=DATA_CSV  $4=IMG_DIR  $5=MASK_DIR
# ─────────────────────────────────────────────────────────────────────────────
run_task() {
    local TASK_LABEL=$1
    local RESUME=$2
    local DATA_CSV=$3
    local IMG_DIR=$4
    local MASK_DIR=$5

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Task: ${TASK_LABEL}"
    echo "  Checkpoint: ${RESUME}"
    echo "  Data: ${DATA_CSV}"
    echo "════════════════════════════════════════════════════════════════════"

    TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py \
        --model          SMP \
        --finetune       "${SEG_CKPT}" \
        --nb_classes     ${NB_CLASSES} \
        --input_size     ${INPUT_SIZE} \
        --data_path      "${DATA_CSV}" \
        --img_dir        "${IMG_DIR}" \
        --thickness_dir  "${MASK_DIR}" \
        --task           "${TASK_LABEL}-ECE" \
        --output_dir     "${OUTPUT_DIR}/output_dir" \
        --log_dir        "${OUTPUT_DIR}/output_logs" \
        --resume         "${RESUME}" \
        --SMPMode        fuse \
        --smp_fuse_mode  weighted_sum \
        --smp_learnable_alpha \
        --smp_alpha      ${ALPHA} \
        --smp_size_match ${SIZE_MATCH} \
        --fusion_dim     ${FUSION_DIM} \
        --align          ${ALIGN} \
        --enc_idx        ${ENC_IDX} \
        --dec_idx        ${DEC_IDX} \
        --smp_classifier ${SMP_CLASSIFIER} \
        --seg_mask \
        --output_mask \
        --num_workers    8 \
        --batch_size     8 \
        --skip_xai
}

# ── IRB2024 tasks — each dataset has its own trained checkpoint ───────────────
for DATASET in DME_binary_all_split AMD_all_split Glaucoma_binary_all_split ERM_all_split; do
    RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-${IRB2024_FUSE_WS_SUFFIX}/checkpoint-best.pth"
    run_task "${DATASET}" "${RESUME}" \
        "${IRB2024_DATA_DIR}/${DATASET}.csv" \
        "${IRB2024_IMG_DIR}" "${IRB2024_MASK_DIR}"
done

# ── Public dataset tasks — each has its own trained checkpoint ────────────────
# Checkpoint pattern: {STUDY}-{OCTDL|CellData}-all-${SEG_CKPT}-OCT-bs16ep...
# (trained directly on public datasets, not transferred from IRB2024)
OCTDL_DME_RESUME="${RESULTS_DIR}/DME_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth"
OCTDL_AMD_RESUME="${RESULTS_DIR}/AMD_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth"
OCTDL_ERM_RESUME="${RESULTS_DIR}/ERM_all-OCTDL-all-${SEG_CKPT}-${PUBLIC_FUSE_WS_SUFFIX}/checkpoint-best.pth"
CELLDATA_RESUME="${RESULTS_DIR}/DME_all-CellData-all-${SEG_CKPT}-${CELLDATA_FUSE_WS_SUFFIX}/checkpoint-best.pth"

run_task "OCTDL_DME" "${OCTDL_DME_RESUME}" \
    "${OCTDL_DATA_DIR}/DME_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

run_task "OCTDL_AMD" "${OCTDL_AMD_RESUME}" \
    "${OCTDL_DATA_DIR}/AMD_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

run_task "OCTDL_ERM" "${OCTDL_ERM_RESUME}" \
    "${OCTDL_DATA_DIR}/ERM_all.csv" "${OCTDL_IMG_DIR}" "${OCTDL_MASK_DIR}"

run_task "CellData" "${CELLDATA_RESUME}" \
    "${CELLDATA_DATA_DIR}/DME_all.csv" "${CELLDATA_IMG_DIR}" "${CELLDATA_MASK_DIR}"

echo ""
echo "All tasks complete. Results written to ${OUTPUT_DIR}/output_dir/*/ece_test.csv"
