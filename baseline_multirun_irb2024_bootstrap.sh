#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SCRIPT=$1 
DATASET=${2:-"AMD_all_split"} #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${3:-"RETFound_mae"}
FINETUNED_MODEL=${4:-"RETFound_mae_natureOCT"}
LR=${5:-"5e-4"}
NUM_CLASS=${6:-"2"}
weight_decay=${7:-"0.05"}
Eval_score=${8:-"default"}
Modality=${9:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${10:-0} # 0, 500, 1000
ADDCMD=${11:-""}
ADDCMD2=${12:-""}

NUM_K=0

#sbatch baseline_multirun_irb2024_bootstrap.sh finetune_retfound_UFbenchmark_irb2024v5_bootstrap.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2 0.05 roc_auc OCT 500 --bootstrap_runs
#sbatch baseline_multirun_irb2024_bootstrap.sh finetune_retfound_UFbenchmark_irb2024v5_bootstrap.sh DR_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 6 0.05 roc_auc OCT 300 --bootstrap_runs
SUBSET_SEEDS=(1 2 3 4 5 6 7 8 9 10)
for i in "${!SUBSET_SEEDS[@]}"
do
    # Create a job name based on the variables
    SUBSETSEED="${SUBSET_SEEDS[$i]}"
    echo "Running dataset: $DATASET with subset_seed=$SUBSETSEED"
    # Submit the job to Slurm
    echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $SUBSETSEED $ADDCMD $ADDCMD2"
    bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM $SUBSETSEED $ADDCMD $ADDCMD2
done