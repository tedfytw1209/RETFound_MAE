#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate octxai

# ============================================================================
# Model Definitions
# ============================================================================


SCRIPT=$1
# Classifier checkpoint to load (main-style --resume)
RESUME_CKPT=$2
# SMP segmentation checkpoint to load into SMPClassifier (main-style --finetune)
SEG_CKPT=${3:-"/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth"}

# SMP hyper-params (follow finetune_retfound_UFbenchmark_irb2024v5_smp.sh conventions)
SMPMode=${4:-"dec"}                 # dec, enc, fuse
SMPFuseMode=${5:-"weighted_sum"}    # weighted_sum/add/channel_merge/channel_multiply/multiply
SMPAlpha=${6:-0.5}                  # 0.0-1.0
SMPSizeMatch=${7:-"decoder_to_encoder"}  # decoder_to_encoder / encoder_to_decoder
FUSION_DIM=${8:-0}
ALIGN=${9:-"pre"}                   # pre / post
ENC_IDX=${10:-"-1"}
DEC_IDX=${11:-"-1"}
SMPClassifier=${12:-"linear"}       # linear / conv

OUTPUT_DIR=${13:-"./heatmap_debug_batch"}
batch_size=${14:-4}

# ============================================================================
# Common parameters
# ============================================================================
DATASET_DIR="/blue/ruogu.fang/tienyuchang/OCT_EDA"
DATASET_FNAME="sampled_labels01.csv"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
target_module="encoder decoder head"

# ============================================================================
# Run XAI for base models (DME_finetuned)
# ============================================================================
python "$SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --dataset_fname "$DATASET_FNAME" \
        --thickness_dir "$THICKNESS_DIR" \
        --thickness_csv "$THICKNESS_CSV" \
        --model "SMP" \
        --finetune "$SEG_CKPT" \
        --resume "$RESUME_CKPT" \
        --SMPMode "$SMPMode" \
        --smp_fuse_mode "$SMPFuseMode" \
        --smp_alpha "$SMPAlpha" \
        --smp_size_match "$SMPSizeMatch" \
        --fusion_dim "$FUSION_DIM" \
        --align "$ALIGN" \
        --enc_idx "$ENC_IDX" \
        --dec_idx "$DEC_IDX" \
        --smp_classifier "$SMPClassifier" \
        --target_module "$target_module" \
        --task DME \
        --num_samples -1 \
        --xai "gradcam hirescam gradcam++" \
        --batch_size $batch_size \
        --input_size 512 \
        --nb_classes 2 \
        --load_mask \
        --draw_layer \
        --output_dir "$OUTPUT_DIR" \
        --verbose