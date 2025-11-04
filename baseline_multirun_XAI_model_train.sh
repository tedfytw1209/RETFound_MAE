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
DATASET=$2 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${3:-"RETFound_mae"}
FINETUNED_MODEL=${4:-"RETFound_mae_natureOCT"}
LR=${5:-"5e-4"}
NUM_CLASS=${6:-"2"}
weight_decay=${7:-"0.05"}
Eval_score=${8:-"default"}
Modality=${9:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${10:-0} # 0, 500, 1000
ADDCMD=${10:-""}
ADDCMD2=${11:-""}

NUM_K=0

#sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_UFbenchmark_irb2024v5.sh DME_binary_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT 0 --add_mask --train_no_aug
#sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_UFbenchmark_irb2024v5_init.sh DME_binary_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT 0 --add_mask --train_no_aug
# Different additional commands to try
#sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_Celldata.sh DME_all RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
#sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_OCTDL.sh DME_all RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
#sbatch baseline_multirun_XAI_model_train.sh finetune_retfound_OCTIDdata.sh DR_all RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 default OCT
MODELS=(timm_efficientnet-b4 resnet-50 vit-base-patch16-224 RETFound_mae)  # List of models
FINETUNED_MODELS=(timm_efficientnet-b4 microsoft/resnet-50 google/vit-base-patch16-224-in21k RETFound_mae_natureOCT)  # Number of classes for each dataset
for i in "${!MODELS[@]}"
do
    #for j in "${!ADDCMDS1[@]}"
    #do
    # Create a job name based on the variables
    MODEL="${MODELS[$i]}"
    FINETUNED_MODEL="${FINETUNED_MODELS[$i]}"
    #ADDCMD="${ADDCMDS1[$j]}"
    #ADDCMD2="${ADDCMDS2[$j]}"
    # Submit the job to Slurm
    echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2"
    bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2
    #done
done
