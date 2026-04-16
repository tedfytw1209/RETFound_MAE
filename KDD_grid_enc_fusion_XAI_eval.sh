#!/bin/bash
# Grid XAI evaluation launcher — submits one SLURM eval job per (ENC_IDX × FUSION_DIM).
# Mirrors KDD_grid_enc_fusion_train.sh parameter combinations for direct comparison.
#
# Usage:
#   bash KDD_grid_enc_fusion_XAI_eval.sh          # evaluate all 2×2 = 4 combinations
#
# Grid axes:
#   ENC_IDX    — encoder layer index (maps to DEC_IDX pos in training script): -1, -3
#   FUSION_DIM — fusion spatial dimension: 16, 32

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
FIXED_DEC_IDX=-1   # fixed DEC_IDX (pos 17 in training script, always -1)
ALPHA_TYPE=scalar
NUM_CLASS=2
INPUT_SIZE=512

ENC_IDXS=(-2)
FUSION_DIMS=(4 16 32)

for ENC_IDX in "${ENC_IDXS[@]}"; do
    for FUSION_DIM in "${FUSION_DIMS[@]}"; do
        RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-${FUSION_DIM}-fea${ENC_IDX}${FIXED_DEC_IDX}-0.5-decoder_to_encoder-conv-alpha${ALPHA_TYPE}---seg_mask---smp_learnable_alpha-/checkpoint-best.pth"

        CMD="sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh ${DATASET} ${NUM_CLASS} SMP ${BASE_CKPT} ${RESUME} ${INPUT_SIZE} fuse weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre ${ENC_IDX} ${FIXED_DEC_IDX} head -1 conv --seg_mask --smp_learnable_alpha --smp_alpha_type ${ALPHA_TYPE}"
        echo "${CMD}"
        #eval "${CMD}"
    done
done

ENC_IDXS=(-1 -3)
FUSION_DIMS=(9)

for ENC_IDX in "${ENC_IDXS[@]}"; do
    for FUSION_DIM in "${FUSION_DIMS[@]}"; do
        RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-${FUSION_DIM}-fea${ENC_IDX}${FIXED_DEC_IDX}-0.5-decoder_to_encoder-conv-alpha${ALPHA_TYPE}---seg_mask---smp_learnable_alpha-/checkpoint-best.pth"

        CMD="sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh ${DATASET} ${NUM_CLASS} SMP ${BASE_CKPT} ${RESUME} ${INPUT_SIZE} fuse weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre ${ENC_IDX} ${FIXED_DEC_IDX} head -1 conv --seg_mask --smp_learnable_alpha --smp_alpha_type ${ALPHA_TYPE}"
        echo "${CMD}"
        #eval "${CMD}"
    done
done
