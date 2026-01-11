#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SCRIPT=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
INPUT_SIZE=${4:-224}
SMPMode=${5:-"dec"} # dec, enc, fuse
SMPFuseMode=${6:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${7:-0.5} # 0.0-1.0
SMPSizeMatch=${8:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${9:-0} # 0 for default
ALIGN=${10:-"pre"} # 0 for default
ENC_IDX=${11:-"-1"} # -1 for last encoder layer
DEC_IDX=${12:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${13:-"encoder"} # encoder, decoder, head
SELECT_INDEX=${14:-"-1"} # -1 for last layer
SMPClassifier=${15:-"linear"} # linear, conv
ADDCMD3=${16:-""}

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split DR_binary_all_split DME_binary_all_split)  # List of datasets
#CLASSES=(2 2 2)  # Number of classes for each dataset
DATASETS=(DME_binary_all_split)  # List of datasets
CLASSES=(2)  # Number of classes for each dataset
STEP_PIXELS=1024
Thickness_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
ADDCMD="--add_mask"
ADDCMD2="--smp_learnable_alpha"

#bash baseline_multirun_XAI_eval_smpmask.sh finetune_retfound_UFbenchmark_v5_eval_smp.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth 512 enc weighted_sum 0.5 encoder_to_decoder 16 -1 -1 encoder -1
#XAI_METHODS=("gradcamv2" "scorecam" "crp")  # List of XAI methods
XAI_METHODS=("gradcamv2" "hirescam" "gradcam++")  # List of XAI methods
#XAI_METHODS=("crp")  # List of XAI methods
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    for XAI in "${XAI_METHODS[@]}"
    do
        # DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0--/
        echo "sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-$SMPMode---add_mask---train_no_aug/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $ADDCMD $ADDCMD2 $ADDCMD3"
        #sbatch $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $MODEL_DIR/$DATASET-IRB2024_v5-all-$FINETUNED_MODEL-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-$SMPMode---add_mask---train_no_aug/checkpoint-best.pth $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $ADDCMD $ADDCMD2 $ADDCMD3
    done
done
