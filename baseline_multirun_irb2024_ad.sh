#! /bin/bash

SCRIPT=$1 #
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"1e-3"}
Epochs=${5:-"100"}
Num_CLASS=${6:-"2"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
ADDCMD=${9:-""}
ADDCMD2=${10:-""}

NUM_K=0

#bash baseline_multirun_irb2024_ad.sh finetune_retfound_adcon_irb2024_imgf.sh RETFound_mae RETFound_mae_natureOCT 5e-4 50 2 default OCT --bal_sampler
#bash baseline_multirun_irb2024_ad.sh finetune_retfound_adcon_irb2024_imgf.sh efficientnet-b4 google/efficientnet-b4 5e-4 50 2 default OCT --bal_sampler
#bash baseline_multirun_irb2024_ad.sh finetune_retfound_adcon_irb2024_imgf.sh vit-base-patch16-224 google/vit-base-patch16-224 5e-4 50 2 default OCT --bal_sampler
#bash baseline_multirun_irb2024_ad.sh finetune_retfound_adcon_irb2024_imgf.sh resnet-50 microsoft/resnet-50 5e-4 50 2 default OCT --bal_sampler
DATASETS=(ad_mci_control_detect_data ad_mci_detect_data mci_control_detect_data)  # List of datasets
CLASSES=(3 2 2)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    # Submit the job to Slurm
    echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $Epochs $NUM_CLASS $Eval_score $Modality $ADDCMD $ADDCMD2"
    sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $LR $Epochs $NUM_CLASS $Eval_score $Modality $ADDCMD $ADDCMD2
done