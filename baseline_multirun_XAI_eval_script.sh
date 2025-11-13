#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SCRIPT=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
INPUT_SIZE=${4:-224}

NUM_K=0
MODEL_DIR="output_dir"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
DATASETS=(DME_all)  # List of datasets
CLASSES=(2)  # Number of classes for each dataset

#bash baseline_multirun_XAI_eval.sh finetune_retfound_UFbenchmark_v5_eval.sh RETFound_mae RETFound_mae_natureOCT 224
#XAI_METHODS=("attn" "gradcam")  # List of XAI methods
#SCRIPTS=("finetune_retfound_Celldata_eval.sh" "finetune_retfound_OCTDL_eval.sh")
#PARAMS=("OCT-bs16ep3lr5e-4optadamw-defaulteval" "OCT-bs16ep50lr5e-4optadamw-defaulteval")
#DATATYPE=("CellData" "OCTDL")
SCRIPTS=("finetune_retfound_Celldata_eval.sh")
PARAMS=("OCT-bs16ep3lr5e-4optadamw-defaulteval")
DATATYPE=("CellData")
XAI_METHODS=("attn" "gradcamv2" "scorecam" "crp")  # List of XAI methods
#MODELS=(timm_efficientnet-b4 resnet-50 vit-base-patch16-224 RETFound_mae)  # List of models
#FINETUNED_MODELS=(timm_efficientnet-b4 microsoft/resnet-50 google/vit-base-patch16-224-in21k RETFound_mae_natureOCT)  # Number of classes for each dataset
MODELS=(timm_efficientnet-b4)  # List of models
FINETUNED_MODELS=(timm_efficientnet-b4)  # Number of classes for each dataset

for s in "${!SCRIPTS[@]}"
do
    SCRIPT="${SCRIPTS[$s]}"
    PARAM="${PARAMS[$s]}"
    DATATYPE="${DATATYPE[$s]}"
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    for j in "${!MODELS[@]}"
    do
        MODEL="${MODELS[$j]}"
        FINETUNED_MODEL="${FINETUNED_MODELS[$j]}"
    for XAI in "${XAI_METHODS[@]}"
    do
        # Submit the job to Slurm
        #output_dir/DME_all-CellData-all-RETFound_mae_natureOCT-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth
        echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-$DATATYPE-all-$FINETUNED_MODEL-$PARAM--/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI"
        #bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-$DATATYPE-all-$FINETUNED_MODEL-$PARAM--/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI
        done
        done
    done
done
