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
RESUME=${4:-""}
INPUT_SIZE=${5:-224}
SMPMode=${6:-"dec"} # dec, enc, fuse
SMPFuseMode=${7:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${8:-0.5} # 0.0-1.0
SMPSizeMatch=${9:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${10:-0} # 0 for default
ALIGN=${11:-"pre"} # 0 for default
ENC_IDX=${12:-"-1"} # -1 for last encoder layer
DEC_IDX=${13:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${14:-"encoder"} # encoder, decoder, head
XAI_METHOD=${15:-"gradcamv2"} # gradcamv2, scorecam, crp, hirescam
SMPClassifier=${16:-"linear"} # linear, conv
ADDCMD=${17:-""}
ADDCMD2=${18:-""}
ADDCMD3=${19:-""}
NUM_K=0
MODEL_DIR="/orange/ruogu.fang/tienyuchang/RETfound_results"
#microsoft/resnet-50, timm_efficientnet-b4, google/vit-base-patch16-224-in21k, RETFound_mae_natureOCT
#DATASETS=(AMD_all_split DR_binary_all_split DME_binary_all_split)  # List of datasets
#CLASSES=(2 2 2)  # Number of classes for each dataset
DATASETS=(DME_binary_all_split)  # List of datasets
CLASSES=(2)  # Number of classes for each dataset
STEP_PIXELS=1024
Thickness_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"

#sbatch baseline_multirun_XAI_eval_smp_ab.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder gradcamv2 conv
#sbatch baseline_multirun_XAI_eval_smp_ab.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-dec-smpweighted_sum-pre-0-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 decoder gradcamv2 conv

#sbatch baseline_multirun_XAI_eval_smp_ab.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth 512 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head gradcamv2 conv --seg_mask --smp_learnable_alpha

#sbatch baseline_multirun_XAI_eval_smp_ab.sh finetune_retfound_UFbenchmark_v5_eval_smp_full.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-fea-1-1-0.5-encoder_to_decoder---seg_mask--/checkpoint-best.pth 512 fuse multiply 0.5 decoder_to_encoder 16 pre -1 -1 encoder gradcamv2 conv --seg_mask

#XAI_METHODS=("gradcamv2" "scorecam" "crp")  # List of XAI methods
#XAI_METHODS=("gradcamv2")  # List of XAI methods
#XAI_METHODS=("hirescam")  # List of XAI methods
#XAI_METHODS=("crp")  # List of XAI methods

if [[ "$TARGET_MODULE" == "encoder" ]]; then
  if [[ "$SMPMode" == "enc" ]]; then
    SELECT_INDEXS=(10 23 42)
  elif [[ "$SMPMode" == "fuse" ]]; then
    SELECT_INDEXS=(23 42)
  else
    SELECT_INDEXS=(10 23 42 52)
  fi
elif [[ "$TARGET_MODULE" == "decoder" ]]; then
  if [[ "$SMPMode" == "dec" ]]; then
    SELECT_INDEXS=(7 9)
  elif [[ "$SMPMode" == "fuse" ]]; then
    SELECT_INDEXS=(7 9)
  else
    SELECT_INDEXS=(1 3 5 7 9)
  fi
elif [[ "$TARGET_MODULE" == "head" ]]; then
  SELECT_INDEXS=(0 1 2)
else
  echo "Invalid target_module: $TARGET_MODULE"
  exit 1
fi

for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "Running dataset: $DATASET with num_class=$NUM_CLASS"
    for SELECT_INDEX in "${SELECT_INDEXS[@]}"
    do
        # Submit the job to Slurm
        echo "bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME $NUM_CLASS $INPUT_SIZE $XAI_METHOD $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ALIGN $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3"
        bash $SCRIPT $DATASET $MODEL $FINETUNED_MODEL $RESUME $NUM_CLASS $INPUT_SIZE $XAI_METHOD $STEP_PIXELS $Thickness_DIR $SMPMode $SMPFuseMode $SMPAlpha $SMPSizeMatch $FUSION_DIM $ALIGN $ENC_IDX $DEC_IDX $TARGET_MODULE $SELECT_INDEX $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3
    done
done
