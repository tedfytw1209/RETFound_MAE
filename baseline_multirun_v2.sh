#! /bin/bash

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_all_split_binary 2, Glaucoma_all_split_binary 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
NUM_K=${7:-"0"}
ADDCMD=${8:-""}
ADDCMD2=${9:-""}

#bash slm/finetune/2025-0606-finetune-UF-Cohorts_v2_split5.sh 5e-3 1e-6 100 dual v3 0.1
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_all_split_binary Glaucoma_all_split_binary)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    # Submit the job to Slurm
    sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $NUM_K "$ADDCMD" "$ADDCMD2"
done