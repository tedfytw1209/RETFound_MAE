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
DATASET=${2:-"DME_binary_all_split"}
NUM_CLASS=${3:-"2"}
MODEL=${4:-"RETFound_mae"}
FINETUNED_MODEL=${5:-"RETFound_mae_natureOCT"}
RESUME=${6:-""}
INPUT_SIZE=${7:-224}
SMPMode=${8:-"dec"} # dec, enc, fuse
SMPFuseMode=${9:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${10:-0.5} # 0.0-1.0
SMPSizeMatch=${11:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${12:-0} # 0 for default
ALIGN=${13:-"pre"} # 0 for default
ENC_IDX=${14:-"-1"} # -1 for last encoder layer
DEC_IDX=${15:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${16:-"encoder"} # encoder, decoder, head
SELECT_INDEX=${17:-"-1"} # -1 for last layer
SMPClassifier=${18:-"linear"} # linear, conv
ADDCMD=${19:-""}
ADDCMD2=${20:-""}
ADDCMD3=${21:-""}

NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split DR_binary_all_split DME_binary_all_split)  # List of datasets
#CLASSES=(2 2 2)  # Number of classes for each dataset
#DATASETS=(DME_binary_all_split)  # List of datasets
#CLASSES=(2)  # Number of classes for each dataset
STEP_PIXELS=1024
Thickness_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"

#sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh DME_binary_all_split 2 SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
#sbatch baseline_multirun_XAI_eval_smp.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh DME_binary_all_split 2 SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 decoder -1 conv

#XAI_METHODS=("gradcamv2" "scorecam" "crp")  # List of XAI methods
XAI_METHODS=("hirescam" "gradcamv2" "gradcam++")  # List of XAI methods
#XAI_METHODS=("hirescam")  # List of XAI methods
#XAI_METHODS=("crp")  # List of XAI methods
for XAI in "${XAI_METHODS[@]}"
do
    # Submit the job to Slurm
    #/orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-{0}-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth
    echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ALIGN $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3"
    bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME $NUM_CLASS $INPUT_SIZE $XAI $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ALIGN $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3
done
