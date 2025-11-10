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

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
#CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
DATASETS=(DME_binary_all_split AMD_binary_all_split)  # List of datasets
CLASSES=(2 2)  # Number of classes for each dataset

#sbatch baseline_multirun_XAI_eval.sh finetune_retfound_UFbenchmark_v5_eval.sh RETFound_mae RETFound_mae_natureOCT
#XAI_METHODS=("attn" "gradcam")  # List of XAI methods
XAI_METHODS=("attn" "gradcamv2" "scorecam" "rise" "crp")  # List of XAI methods
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    for XAI in "${XAI_METHODS[@]}"
    do
        # Submit the job to Slurm
        echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0---add_mask---train_no_aug/checkpoint-best.pth $NUM_CLASS $XAI"
        sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0---add_mask---train_no_aug/checkpoint-best.pth $NUM_CLASS $XAI
    done
done
