#! /bin/bash

SCRIPT=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"auc"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${9:-0} # 0, 500, 1000

NUM_K=0

# sbatch baseline_multirun_irb2024_fairness.sh infernce_retfound_UFirb2024v5_fairness.sh RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 auc OCT 0
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
SUBGROUP_COLS=("race_ethnicity" "race_ethnicity" "gender_source_value" "age_group" "age_group") # age_group, race_ethnicity, gender_source_value
PROTECTED_VALUES=("NHB" "HISPANIC" "FEMALE" "lt45" "65-74" "ge75")
PRIVALENT_VALUES=("NHW" "NHW" "MALE" "45-64" "45-64" "45-64")

for i in "${!DATASETS[@]}"
do
    for j in "${!SUBGROUP_COLS[@]}"
    do
    SUBGROUP_COL="${SUBGROUP_COLS[$j]}"
    PROTECTED_VALUE="${PROTECTED_VALUES[$j]}"
    PRIVALENT_VALUE="${PRIVALENT_VALUES[$j]}"
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    RESUME_MODEL=/orange/ruogu.fang/tienyuchang/RETfound_results/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-$Modality-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth
    # Submit the job to Slurm
    echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $SUBGROUP_COL $PROTECTED_VALUE $PRIVALENT_VALUE"
    bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $SUBGROUP_COL $PROTECTED_VALUE $PRIVALENT_VALUE
    done
done