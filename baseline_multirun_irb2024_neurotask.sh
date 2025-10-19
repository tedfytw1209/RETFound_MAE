#! /bin/bash

SCRIPT=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${9:-0} # 0, 500, 1000
ADDCMD=${10:-""}
ADDCMD2=${11:-""}

NUM_K=0

#bash baseline_multirun_irb2024_neurotask.sh finetune_retfound_adcon_irb2024_v5.sh RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 roc_auc OCT
DATASETS=(ad_control_detect_data ad_mci_detect_data mci_control_detect_data ad_mci_control_detect_data) 
CLASSES=(2 2 2 3)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    # Submit the job to Slurm
    echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2"
    sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $ADDCMD $ADDCMD2
done