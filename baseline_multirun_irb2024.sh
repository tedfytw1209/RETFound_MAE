#! /bin/bash

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
ADDCMD=${9:-""}
ADDCMD2=${10:-""}

NUM_K=0

#bash baseline_multirun_irb2024.sh finetune_retfound_UFbenchmark_irb2024v3.sh RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 mcc OCT --testval
#bash baseline_multirun_irb2024.sh finetune_retfound_UFbenchmark_pytorchvit.sh pytorchvit B_16_imagenet1k 0.01 2 1e-6 mcc OCT --testval
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    # Submit the job to Slurm
    echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $ADDCMD $ADDCMD2"
    sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $ADDCMD $ADDCMD2
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
done