#!/bin/bash
# Transfer evaluation launcher — submits SLURM eval jobs for BASELINE models trained on UF dataset,
# evaluated on public datasets (OCTDL, CellData).
#
# Baseline models: RETFound_mae, timm_efficientnet-b4, vit-base-patch16-224, resnet-50
#
# Usage:
#   bash KDD_UF_transfer_public_baselines.sh

RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
NUM_CLASS=2
INPUT_SIZE=224
STEP_PIXELS=224
XAI_METHOD=${1:-"hirescam"}  # Default to "hirescam" if not provided

# Public dataset study names (equivalent disease split used in transfer eval)
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

#XAI_METHODS=("hirescam" "gradcamv2" "gradcam++")  # List of XAI methods

# Baseline model pairs: MODEL (architecture) and FINETUNED_MODEL (checkpoint identifier)
MODELS=(
    "RETFound_mae"
    "timm_efficientnet-b4"
    "vit-base-patch16-224"
    "resnet-50"
)
FINETUNED_MODELS=(
    "RETFound_mae_natureOCT"
    "timm_efficientnet-b4"
    "google/vit-base-patch16-224-in21k"
    "microsoft/resnet-50"
)

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    FINETUNED_MODEL="${FINETUNED_MODELS[$i]}"

    RESUME="${RESULTS_DIR}/${DATASET}-${DATA_TYPE}-all-${FINETUNED_MODEL}-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0---/checkpoint-best.pth"
    echo "${RESUME}"

    # --- Eval on OCTDL ---
    echo "Submitting OCTDL eval for MODEL=${MODEL} ..."
    bash finetune_retfound_OCTDL_eval.sh ${OCTDL_STUDY} ${MODEL} ${FINETUNED_MODEL} "${RESUME}" ${NUM_CLASS} ${INPUT_SIZE} ${XAI_METHOD} ${STEP_PIXELS} /orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

    # --- Eval on CellData ---
    echo "Submitting CellData eval for MODEL=${MODEL} ..."
    bash finetune_retfound_Celldata_eval.sh ${CELLDATA_STUDY} ${MODEL} ${FINETUNED_MODEL} "${RESUME}" ${NUM_CLASS} ${INPUT_SIZE} ${XAI_METHOD} ${STEP_PIXELS} /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
done
