#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# UF -> public-dataset transfer eval, baseline architectures.
# One UF-trained checkpoint per (task, architecture); XAI eval run on the
# equivalent-disease OCTDL split, and on CellData (OCT2017) where it exists
# (CellData only has a DME split).
#
# Table (Task / Source / Target / RETFound / ViT / EfficientNet-b4 / Resnet-50):
#   DME  UF DME -> OCTDL DME
#   DME  UF DME -> OCT2017(CellData) DME
#   AMD  UF AMD -> OCTDL AMD
#   ERM  UF ERM -> OCTDL ERM
# RETFound cells are already done -> commented out with #. Uncomment to rerun.
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=50
NUM_CLASS=2
XAI_METHOD=${1:-"hirescam"}  # not used — finetune_retfound_OCTDL_eval.sh / finetune_retfound_Celldata_eval.sh loop their own XAI methods internally
ADD_WORD1=""
ADD_WORD2=""

OCTDL_MASK_DIR=/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/
CELLDATA_MASK_DIR=/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/

# Baseline model pairs: MODEL (architecture) / FINETUNED_MODEL (checkpoint id) / INPUT_SIZE
# RETFound_mae         RETFound_mae_natureOCT              224
# vit-base-patch16-224 google/vit-base-patch16-224-in21k   224
# timm_efficientnet-b4 timm_efficientnet-b4                380
# resnet-50            microsoft/resnet-50                 224

octdl_eval() {
    # $1=STUDY $2=MODEL $3=FINETUNED_MODEL $4=RESUME $5=INPUT_SIZE
    bash finetune_retfound_OCTDL_eval.sh "$1" "$2" "$3" "$4" ${NUM_CLASS} "$5" ${XAI_METHOD} "$5" ${OCTDL_MASK_DIR} enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
}

celldata_eval() {
    # $1=STUDY $2=MODEL $3=FINETUNED_MODEL $4=RESUME $5=INPUT_SIZE
    bash finetune_retfound_Celldata_eval.sh "$1" "$2" "$3" "$4" ${NUM_CLASS} "$5" ${XAI_METHOD} "$5" ${CELLDATA_MASK_DIR} enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
}

uf_resume() {
    # $1=DATASET(UF) $2=FINETUNED_MODEL
    echo "${RESULTS_DIR}/$1-${DATA_TYPE}-all-$2-${MODALITY}-bs${BATCH_SIZE}ep${EPOCHS}lr${LR}optadamw-defaulteval-trsub0-${ADD_WORD1}-${ADD_WORD2}/checkpoint-best.pth"
}

# ════════════════════════════════════════════════════════════════════════════
# Task: DME   (Source: UF DME  ->  Target: OCTDL DME / OCT2017(CellData) DME)
# ════════════════════════════════════════════════════════════════════════════
DATASET=DME_binary_all_split
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

RESUME_RETFound=$(uf_resume ${DATASET} RETFound_mae_natureOCT)
RESUME_ViT=$(uf_resume ${DATASET} google/vit-base-patch16-224-in21k)
RESUME_EffNet=$(uf_resume ${DATASET} timm_efficientnet-b4)
RESUME_Resnet=$(uf_resume ${DATASET} microsoft/resnet-50)

# RETFound — done
# octdl_eval ${OCTDL_STUDY} RETFound_mae RETFound_mae_natureOCT "${RESUME_RETFound}" 224
# celldata_eval ${CELLDATA_STUDY} RETFound_mae RETFound_mae_natureOCT "${RESUME_RETFound}" 224

#octdl_eval ${OCTDL_STUDY} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_ViT}" 224
#celldata_eval ${CELLDATA_STUDY} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_ViT}" 224

#octdl_eval ${OCTDL_STUDY} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_EffNet}" 380
#celldata_eval ${CELLDATA_STUDY} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_EffNet}" 380

#octdl_eval ${OCTDL_STUDY} resnet-50 microsoft/resnet-50 "${RESUME_Resnet}" 224
#celldata_eval ${CELLDATA_STUDY} resnet-50 microsoft/resnet-50 "${RESUME_Resnet}" 224

# ════════════════════════════════════════════════════════════════════════════
# Task: AMD   (Source: UF AMD  ->  Target: OCTDL AMD)
# ════════════════════════════════════════════════════════════════════════════
DATASET=AMD_all_split
OCTDL_STUDY=AMD_all

RESUME_RETFound=$(uf_resume ${DATASET} RETFound_mae_natureOCT)
RESUME_ViT=$(uf_resume ${DATASET} google/vit-base-patch16-224-in21k)
RESUME_EffNet=$(uf_resume ${DATASET} timm_efficientnet-b4)
RESUME_Resnet=$(uf_resume ${DATASET} microsoft/resnet-50)

# RETFound — done
# octdl_eval ${OCTDL_STUDY} RETFound_mae RETFound_mae_natureOCT "${RESUME_RETFound}" 224

octdl_eval ${OCTDL_STUDY} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_ViT}" 224
octdl_eval ${OCTDL_STUDY} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_EffNet}" 380
octdl_eval ${OCTDL_STUDY} resnet-50 microsoft/resnet-50 "${RESUME_Resnet}" 224

# ════════════════════════════════════════════════════════════════════════════
# Task: ERM   (Source: UF ERM  ->  Target: OCTDL ERM)
# ════════════════════════════════════════════════════════════════════════════
DATASET=ERM_all_split
OCTDL_STUDY=ERM_all

RESUME_RETFound=$(uf_resume ${DATASET} RETFound_mae_natureOCT)
RESUME_ViT=$(uf_resume ${DATASET} google/vit-base-patch16-224-in21k)
RESUME_EffNet=$(uf_resume ${DATASET} timm_efficientnet-b4)
RESUME_Resnet=$(uf_resume ${DATASET} microsoft/resnet-50)

# RETFound — done
# octdl_eval ${OCTDL_STUDY} RETFound_mae RETFound_mae_natureOCT "${RESUME_RETFound}" 224

octdl_eval ${OCTDL_STUDY} vit-base-patch16-224 google/vit-base-patch16-224-in21k "${RESUME_ViT}" 224
octdl_eval ${OCTDL_STUDY} timm_efficientnet-b4 timm_efficientnet-b4 "${RESUME_EffNet}" 380
octdl_eval ${OCTDL_STUDY} resnet-50 microsoft/resnet-50 "${RESUME_Resnet}" 224
