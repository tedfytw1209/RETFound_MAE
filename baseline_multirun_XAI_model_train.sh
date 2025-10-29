#! /bin/bash

DATASET=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
NUM_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${9:-0} # 0, 500, 1000
ADDCMD="--add_mask"
ADDCMD2="--train_no_aug"

NUM_K=0

#bash baseline_multirun_XAI_model_train.sh DME_binary_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
MODELS=(timm_efficientnet-b4 resnet-50 vit-base-patch16-224 RETFound_mae)  # List of models
FINETUNED_MODELS=(timm_efficientnet-b4 microsoft/resnet-50 google/vit-base-patch16-224-in21k RETFound_mae_natureOCT)  # Number of classes for each dataset
for i in "${!MODELS[@]}"
do
    # Create a job name based on the variables
    MODEL="${MODELS[$i]}"
    FINETUNED_MODEL="${FINETUNED_MODELS[$i]}"
    # Submit the job to Slurm
    echo "sbatch finetune_retfound_UFbenchmark_irb2024v5_tmp.sh $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2"
    sbatch finetune_retfound_UFbenchmark_irb2024v5_tmp.sh $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2
done
