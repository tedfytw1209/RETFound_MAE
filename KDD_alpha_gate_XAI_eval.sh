#!/bin/bash
# Alpha-gate XAI evaluation launcher — submits one SLURM eval job per gate type.
# Mirrors KDD_alpha_gate_train.sh parameter combinations for direct comparison.
#
# Usage:
#   bash KDD_alpha_gate_XAI_eval.sh                     # evaluate all 4 gate types
#   bash KDD_alpha_gate_XAI_eval.sh scalar attn          # evaluate specific types only
#
# Gate types: channel  spatial  se  attn

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
FUSION_DIM=9
ENC_IDX=-2
DEC_IDX=-1
NUM_CLASS=2
INPUT_SIZE=512

if [ $# -eq 0 ]; then
    ALPHA_TYPES=(channel spatial se attn)
else
    ALPHA_TYPES=("$@")
fi

for ALPHA_TYPE in "${ALPHA_TYPES[@]}"; do
    RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${BASE_CKPT}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-${FUSION_DIM}-fea${ENC_IDX}${DEC_IDX}-0.5-decoder_to_encoder-conv-alpha${ALPHA_TYPE}---seg_mask---smp_learnable_alpha--/checkpoint-best.pth"

    echo "Submitting XAI eval for alpha_type=${ALPHA_TYPE} ..."
    sbatch baseline_multirun_XAI_eval_smp.sh \
        finetune_retfound_UFbenchmark_v5_eval_smp_full.sh \
        ${DATASET} \
        ${NUM_CLASS} \
        SMP \
        ${BASE_CKPT} \
        "${RESUME}" \
        ${INPUT_SIZE} \
        fuse \
        weighted_sum \
        0.5 \
        decoder_to_encoder \
        ${FUSION_DIM} \
        pre \
        ${ENC_IDX} \
        ${DEC_IDX} \
        head \
        -1 \
        conv \
        "--seg_mask" \
        "--smp_learnable_alpha" \
        "--smp_alpha_type ${ALPHA_TYPE}"
done
