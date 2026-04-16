#!/bin/bash
# XAI evaluation launcher for random-init segmentation models.
# Mirrors KDD_randinit_seg_train.sh configurations for direct comparison.
#
# Usage:
#   bash KDD_randinit_seg_XAI_eval.sh
#
# Configs:
#   1) fuse mode — FUSION_DIM=9, ENC_IDX=-2, --seg_mask --smp_learnable_alpha
#   2) enc  mode — FUSION_DIM=0, ENC_IDX=-1, no seg_mask/learnable_alpha

BASE_CKPT=na
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
FIXED_DEC_IDX=-1
ALPHA_TYPE=scalar
NUM_CLASS=2
INPUT_SIZE=512

# ── Config 1: fuse, FUSION_DIM=9, ENC_IDX=-2 ────────────────────────────────
ENC_IDX=-2
FUSION_DIM=9
RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-${FUSION_DIM}-fea${ENC_IDX}${FIXED_DEC_IDX}-0.5-decoder_to_encoder-conv-alpha${ALPHA_TYPE}---seg_mask---smp_learnable_alpha-/checkpoint-best.pth"

CMD="sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh ${DATASET} ${NUM_CLASS} SMP ${BASE_CKPT} ${RESUME} ${INPUT_SIZE} fuse weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre ${ENC_IDX} ${FIXED_DEC_IDX} head -1 conv --seg_mask --smp_learnable_alpha --smp_alpha_type ${ALPHA_TYPE}"
echo "Submitting XAI eval: fuse FUSION_DIM=${FUSION_DIM}, ENC_IDX=${ENC_IDX} ..."
echo "${CMD}"
eval "${CMD}"

# ── Config 2: enc, FUSION_DIM=0, ENC_IDX=-1 ─────────────────────────────────
#ENC_IDX=-1
#FUSION_DIM=0
#RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-$#{FUSION_DIM}-fea${ENC_IDX}${FIXED_DEC_IDX}-0.5-decoder_to_encoder-conv-alpha${ALPHA_TYPE}----/checkpoint-best.pth"
#
#CMD="sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh ${DATASET} ${NUM_CLASS} SMP ${BASE_CKPT} ${RESUME} ${INPUT_SIZE} enc #weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre ${ENC_IDX} ${FIXED_DEC_IDX} encoder -1 conv   "
#echo "Submitting XAI eval: enc FUSION_DIM=${FUSION_DIM}, ENC_IDX=${ENC_IDX} ..."
#echo "${CMD}"
#eval "${CMD}"
