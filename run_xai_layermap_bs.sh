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

SCRIPT="case_study_SMP_layermap_bs.py"
STUDY="DME"
MODEL=${1:-"RETFound_mae"}
FINETUNED_MODEL=${2:-"RETFound_mae_natureOCT"}
RESUME=${3:-""}
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
ADDCMD=${16:-""}
ADDCMD2=${17:-""}
ADDCMD3=${18:-""}

OUTPUT_DIR="./heatmap_params_tuning"

# ============================================================================
# Common parameters
# ============================================================================
DATASET_DIR="/blue/ruogu.fang/tienyuchang/OCT_EDA"
DATASET_FNAME="sampled_labels01.csv"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
BATCH_SIZE=4
NUM_SAMPLES=100
Num_CLASS=2
STEP_PIXELS=1024
#XAI_METHODS="gradcamv2 scorecam crp"
XAI_METHODS="gradcamv2 hirescam gradcam++"
#XAI_METHODS="hirescam"

#sbatch run_xai_layermap_bs.sh SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-\{0\}-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

# ============================================================================
# Run XAI for base models (DME_finetuned)
# ============================================================================
python "$SCRIPT" \
        --dataset_dir "$DATASET_DIR" \
        --dataset_fname "$DATASET_FNAME" \
        --thickness_dir "$THICKNESS_DIR" \
        --thickness_csv "$THICKNESS_CSV" \
        --model "$MODEL" \
        --finetune "$FINETUNED_MODEL" \
        --resume "$RESUME" \
        --SMPMode "$SMPMode" \
        --smp_fuse_mode "$SMPFuseMode" \
        --smp_alpha "$SMPAlpha" \
        --smp_size_match "$SMPSizeMatch" \
        --fusion_dim "$FUSION_DIM" \
        --align "$ALIGN" \
        --enc_idx "$ENC_IDX" \
        --dec_idx "$DEC_IDX" \
        --smp_classifier "$SMPClassifier" \
        --target_module "$TARGET_MODULE" \
        --select_index "$SELECT_INDEX" \
        --task "$STUDY" \
        --num_samples $NUM_SAMPLES \
        --xai "$XAI_METHODS" \
        --batch_size $BATCH_SIZE \
        --input_size $INPUT_SIZE \
        --nb_classes "$Num_CLASS" \
        --load_mask \
        --draw_layer \
        --output_dir "$OUTPUT_DIR" \
        $ADDCMD $ADDCMD2 $ADDCMD3