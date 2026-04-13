#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# KDD_computation_eval.sh
#
# Measures parameter / FLOPs / inference-time overhead for:
#   - Baseline models: RETFound_mae, ViT-Base-patch16-224, ResNet-50,
#                      EfficientNet-B4   (from run_xai_layermap_multirun.sh)
#   - Proposed SMP models: enc / dec / fuse
#     (grid mirrors KDD_grid_enc_fusion_train.sh: enc_idxs × fusion_dims)
#
# Pretrained weights are loaded exactly as in main_XAI_evaluation.py:
#   - RETFound_mae : HuggingFace hub  (YukunZhou/RETFound_mae_natureOCT)
#   - ViT-Base     : from_pretrained('google/vit-base-patch16-224-in21k')
#   - ResNet-50    : from_pretrained('microsoft/resnet-50')
#   - EffNetB4     : timm pretrained=True
#   - SMP          : pretrained seg checkpoint  (+ optional resume per config)
#
# Overhead columns (+params_M, +flops_G, +time_ms) are relative to SMP-enc.
#
# Output CSV: ${OUTPUT_CSV}
#
# Usage (interactive):  bash KDD_computation_eval.sh
# Usage (SLURM):        sbatch KDD_computation_eval.sh
# ─────────────────────────────────────────────────────────────────────────────

date; hostname; pwd

module load conda
conda activate octxai

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results

SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
OUTPUT_CSV=/orange/ruogu.fang/tienyuchang/RETfound_results/KDD_computation_overhead.csv

cd /blue/ruogu.fang/tienyuchang/RETFound_MAE || exit 1

# ── SMP architecture (must match training config) ─────────────────────────────
SEG_ARCH=Unet
ENCODER_NAME=resnet50
IN_CHANNELS=3
NB_CLASSES=2
SEG_CLASSES=9
SEG_ACTIVATION=softmax
INPUT_SIZE=512
SMP_FUSE_MODE=weighted_sum
ALPHA=0.5
SIZE_MATCH=decoder_to_encoder
SMP_CLASSIFIER=conv
ALIGN=pre

# ── Baseline pretrained model IDs (matching run_xai_layermap_multirun.sh) ─────
RETFOUND_FINETUNE=RETFound_mae_natureOCT
VIT_FINETUNE=google/vit-base-patch16-224-in21k
RESNET_FINETUNE=microsoft/resnet-50
# EfficientNet-B4 uses timm pretrained=True automatically

# ── Optional finetuned checkpoints for baselines (from run_xai_layermap_multirun.sh)
# Leave empty to use pretrained backbone weights only (sufficient for params/FLOPs/timing).
RETFOUND_RESUME=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth
VIT_RESUME=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth
RESNET_RESUME=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-microsoft/resnet-50-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth
EFFNET_RESUME=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-timm_efficientnet-b4-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth

# ── Optional finetuned checkpoints for the 4 fixed SMP configs ────────────────
# Order: SMP-enc, SMP-dec, SMP-fuse-weighted_sum, SMP-fuse-multiply
SMP_RESUME_ENC=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth
SMP_RESUME_DEC=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth
SMP_RESUME_FUSE_WS=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth
SMP_RESUME_FUSE_MUL=${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${SEG_CKPT}-OCT-bs16ep100lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--/checkpoint-best.pth

# ── Optional grid sweep (appended to fixed configs when --grid is passed) ────
# Mirrors KDD_grid_enc_fusion_train.sh axes.
# Remove "--grid" below to run only the 4 fixed configs from run_xai_layermap_multirun.sh.
GRID_FLAG="--grid"
ENC_IDXS="-1 -2 -3"
DEC_IDXS="-1"
FUSION_DIMS="4 9 16 32"

# ── Baseline models (from run_xai_layermap_multirun.sh) ───────────────────────
BASELINES="RETFound_mae ViT-Base-patch16-224 ResNet-50 EfficientNet-B4"

# ── Timing ───────────────────────────────────────────────────────────────────
N_WARMUP=10
N_RUNS=50
BATCH_SIZE=1    # single-sample inference (production scenario)
DEVICE=cuda

# ─────────────────────────────────────────────────────────────────────────────
python KDD_computation_eval.py \
    --seg_arch          ${SEG_ARCH} \
    --encoder_name      ${ENCODER_NAME} \
    --in_channels       ${IN_CHANNELS} \
    --nb_classes        ${NB_CLASSES} \
    --seg_classes       ${SEG_CLASSES} \
    --seg_activation    ${SEG_ACTIVATION} \
    --input_size        ${INPUT_SIZE} \
    --smp_fuse_mode     ${SMP_FUSE_MODE} \
    --smp_learnable_alpha \
    --alpha             ${ALPHA} \
    --size_match        ${SIZE_MATCH} \
    --use_mask \
    --smp_classifier    ${SMP_CLASSIFIER} \
    --align             ${ALIGN} \
    --seg_ckpt          "${SEG_CKPT}" \
    --retfound_finetune ${RETFOUND_FINETUNE} \
    --retfound_resume   "${RETFOUND_RESUME}" \
    --vit_finetune      "${VIT_FINETUNE}" \
    --vit_resume        "${VIT_RESUME}" \
    --resnet_finetune   "${RESNET_FINETUNE}" \
    --resnet_resume     "${RESNET_RESUME}" \
    --effnet_resume     "${EFFNET_RESUME}" \
    --smp_resumes       "${SMP_RESUME_ENC}" "${SMP_RESUME_DEC}" \
                        "${SMP_RESUME_FUSE_WS}" "${SMP_RESUME_FUSE_MUL}" \
    --enc_idxs          ${ENC_IDXS} \
    --dec_idxs          ${DEC_IDXS} \
    --fusion_dims       ${FUSION_DIMS} \
    --baselines         ${BASELINES} \
    ${GRID_FLAG} \
    --n_warmup          ${N_WARMUP} \
    --n_runs            ${N_RUNS} \
    --batch_size        ${BATCH_SIZE} \
    --device            ${DEVICE} \
    --output_csv        "${OUTPUT_CSV}"
