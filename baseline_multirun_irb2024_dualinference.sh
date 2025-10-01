#! /bin/bash

SCRIPT=$1
MODEL="DualViT"
FINETUNED_MODEL="DualViT_natureOCT"
LR="5e-4"
Num_CLASS="2"
weight_decay="0.05"
Eval_score="default"
Modality="OCT" # CFP, OCT, OCT_CFP
SUBSETNUM="500" # 0, 500, 1000

NUM_K=0

#bash baseline_multirun_irb2024_dualinference.sh infernce_retfound_UFirb2024v5_dualvit.sh
DATASETS=(AMD_all_split Cataract_all_split DR_all_split Glaucoma_all_split DR_binary_all_split Glaucoma_binary_all_split DME_all_split CSR_all_split Drusen_all_split ERM_all_split MH_all_split CRVO_CRAO_all_split PVD_all_split RNV_all_split DME_binary_all_split) 
CLASSES=(2 2 6 6 2 2 5 2 2 2 2 2 2 2 2)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    # sbatch infernce_retfound_UFirb2024v5_dualvit.sh AMD_all_split output_dir/AMD_all_split-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval-trsub500/checkpoint-best.pth output_dir/AMD_all_split-IRB2024_v5-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval-trsub500--/checkpoint-best.pth
    echo "sbatch $SCRIPT $DATASET output_dir/$DATASET-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval-trsub500/checkpoint-best.pth output_dir/$DATASET-IRB2024_v5-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval-trsub500--/checkpoint-best.pth $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM"

    #sbatch $SCRIPT $DATASET output_dir/$DATASET-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval-trsub500--/checkpoint-best.pth output_dir/$DATASET-IRB2024_v5-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval-trsub500--/checkpoint-best.pth $LR $NUM_CLASS $weight_decay $Eval_score $Modality $SUBSETNUM
    
done