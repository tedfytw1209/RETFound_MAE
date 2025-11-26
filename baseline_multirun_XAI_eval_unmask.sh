#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SCRIPT=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
INPUT_SIZE=${4:-224}
STEP_PIXELS=${5:-224}

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split DR_binary_all_split DME_binary_all_split)  # List of datasets
#CLASSES=(2 2 2)  # Number of classes for each dataset
DATASETS=(DME_binary_all_split)  # List of datasets
CLASSES=(2)  # Number of classes for each dataset
ADD_WORDS=""

#sbatch baseline_multirun_XAI_eval.sh finetune_retfound_UFbenchmark_v5_eval.sh RETFound_mae RETFound_mae_natureOCT 224 224
#XAI_METHODS=("hirescam" "gradcam++")  # List of XAI methods
XAI_METHODS=("attn" "gradcamv2" "scorecam" "crp")  # List of XAI methods
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    for XAI in "${XAI_METHODS[@]}"
    do
        # /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-timm_efficientnet-b4-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/
        echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $ADD_WORDS"
        bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $ADD_WORDS
    done
done
