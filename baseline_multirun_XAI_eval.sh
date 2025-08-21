#! /bin/bash

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}

NUM_K=0

#output_dir/AMD_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split)  # List of datasets
CLASSES=(2 2 6 6 2 2)  # Number of classes for each dataset
#DATASETS=(AMD_all_split DR_all_split Glaucoma_all_split)  # List of datasets
#CLASSES=(2 6 6)  # Number of classes for each dataset

#bash baseline_multirun_XAI_eval.sh finetune_retfound_UFbenchmark_v2_eval.sh
#XAI_METHODS=("attn" "gradcam")  # List of XAI methods
XAI_METHODS=("gradcam")  # List of XAI methods
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    for XAI in "${XAI_METHODS[@]}"
    do
        # Submit the job to Slurm
        echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL output_dir/$DATASET-IRB2024_v4-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-5optadamw-roc_auceval--/checkpoint-best.pth $NUM_CLASS $XAI"
        sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL output_dir/$DATASET-IRB2024_v4-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-5optadamw-roc_auceval--/checkpoint-best.pth $NUM_CLASS $XAI
    done
done
