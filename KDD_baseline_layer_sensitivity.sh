#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

EVAL_SCRIPT="finetune_retfound_UFbenchmark_v5_eval_full2.sh"
MODEL=${1:-"RETFound_mae"}
FINETUNED_MODEL=${2:-"RETFound_mae_natureOCT"}
INPUT_SIZE=${3:-"224"}

XAI_METHODS=("hirescam" "gradcam++" "gradcamv2")

MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
DATASET="DME_binary_all_split"
NUM_CLASS=2
ADD_WORD1=""
ADD_WORD2=""
data_type="IRB2024_v5_all"

LAYER_IDXS=(-2 -3)

MODELS=(
  "RETFound_mae RETFound_mae_natureOCT 224"
  "vit-base-patch16-224 google/vit-base-patch16-224-in21k 224"
  "timm_efficientnet-b4 timm_efficientnet-b4 380"
  "resnet-50 microsoft/resnet-50 224"
)

#sbatch KDD_baseline_layer_sensitivity.sh RETFound_mae RETFound_mae_natureOCT 224
#sbatch KDD_baseline_layer_sensitivity.sh vit-base-patch16-224 google/vit-base-patch16-224-in21k 224
#sbatch KDD_baseline_layer_sensitivity.sh timm_efficientnet-b4 timm_efficientnet-b4 380
#sbatch KDD_baseline_layer_sensitivity.sh resnet-50 microsoft/resnet-50 224
for LAYER_IDX in "${LAYER_IDXS[@]}"; do
    CKPT="$MODEL_DIR/$DATASET-$data_type-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0-$ADD_WORD1-$ADD_WORD2/checkpoint-best.pth"
    for XAI_METHOD in "${XAI_METHODS[@]}"; do
        echo "bash $EVAL_SCRIPT $DATASET $MODEL $FINETUNED_MODEL $CKPT $NUM_CLASS $INPUT_SIZE $XAI_METHOD 224 --select_index $LAYER_IDX"
        bash $EVAL_SCRIPT $DATASET $MODEL $FINETUNED_MODEL $CKPT $NUM_CLASS $INPUT_SIZE $XAI_METHOD 224 --select_index $LAYER_IDX
    done
done

echo "=== Sensitivity sweep complete ==="
