#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SCRIPT=$1
DATASET=${2:-"DME_binary_all_split"}
NUM_CLASS=${3:-2}
MODEL=${4:-"RETFound_mae"}
FINETUNED_MODEL=${5:-"RETFound_mae_natureOCT"}
INPUT_SIZE=${6:-224}
ADD_WORD1=${7:-""}
ADD_WORD2=${8:-""}

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
DATASETS=(AMD_all_split Glaucoma_binary_all_split ERM_all_split)  # List of datasets
CLASSES=(2 2 2)  # Number of classes for each dataset
#DATASETS=(DME_binary_all_split)  # List of datasets
#CLASSES=(2)  # Number of classes for each dataset
data_type="IRB2024_v5_all"

#sbatch baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full.sh DME_binary_all_split 2 RETFound_mae RETFound_mae_natureOCT 224
#sbatch baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full.sh DME_binary_all_split 2 resnet-50 microsoft/resnet-50 224
#sbatch baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full.sh DME_binary_all_split 2 vit-base-patch16-224 google/vit-base-patch16-224-in21k 224
#sbatch baseline_multirun_XAI_eval_full.sh finetune_retfound_UFbenchmark_v5_eval_full.sh DME_binary_all_split 2 timm_efficientnet-b4 timm_efficientnet-b4 380
XAI_METHODS=("gradcamv2" "crp")
#XAI_METHODS=("hirescam" "gradcam++" "gradcamv2" "crp")  # List of XAI methods
#XAI_METHODS=("attn" "gradcamv2" "scorecam" "rise" "crp")  # List of XAI methods
#for i in "${!DATASETS[@]}"
#do
#    # Create a job name based on the variables
#    DATASET="${DATASETS[$i]}"
#    NUM_CLASS="${CLASSES[$i]}"
#    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
for XAI in "${XAI_METHODS[@]}"
do
    # Submit the job to Slurm
    echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-$data_type-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0-$ADD_WORD1-$ADD_WORD2/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $INPUT_SIZE $ADD_WORD1 $ADD_WORD2"
    bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-$data_type-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0-$ADD_WORD1-$ADD_WORD2/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $INPUT_SIZE $ADD_WORD1 $ADD_WORD2
done
#done
