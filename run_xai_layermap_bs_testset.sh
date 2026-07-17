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
# Full-test-set variant of run_xai_layermap_bs.sh: instead of the hand-picked
# "<task>_sampled" folder, this passes --data_path/--img_dir/--use_split test so
# case_study_SMP_layermap_bs.py loads the full test split via util.datasets.build_dataset()
# (the same function main_XAI_evaluation.py uses) -- see its _load_full_split_data().

SCRIPT="case_study_SMP_layermap_bs.py"
STUDY=${1:-"DME"} # task label, used for --task / output dir naming (kept independent of the CSV filename)
STUDY_NAME=${2:-"DME_binary_all_split"} # full test-split CSV base filename (no .csv), e.g. DME_binary_all_split, AMD_all_split
MODEL=${3:-"RETFound_mae"}
FINETUNED_MODEL=${4:-"RETFound_mae_natureOCT"}
RESUME=${5:-""}
INPUT_SIZE=${6:-224}
SMPMode=${7:-"dec"} # dec, enc, fuse
SMPFuseMode=${8:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${9:-0.5} # 0.0-1.0
SMPSizeMatch=${10:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${11:-0} # 0 for default
ALIGN=${12:-"pre"} # 0 for default
ENC_IDX=${13:-"-1"} # -1 for last encoder layer
DEC_IDX=${14:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${15:-"encoder"} # encoder, decoder, head
SELECT_INDEX=${16:-"-1"} # -1 for last layer
SMPClassifier=${17:-"linear"} # linear, conv
ADDCMD=${18:-""}
ADDCMD2=${19:-""}
ADDCMD3=${20:-""}

OUTPUT_DIR="./heatmap_params_UF_full"

# ============================================================================
# Common parameters (full UF-cohort test split)
# ============================================================================
DATA_TYPE="IRB2024_v5_all"
DATA_PATH="/orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${DATA_TYPE}/split/tune8-eval2/${STUDY_NAME}.csv"
IMG_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
THICKNESS_CSV="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/thickness_map.csv"
BATCH_SIZE=8
NUM_SAMPLES=-1 # -1 = every image in the test split (set >0 to still sub-sample for a quick look)
Num_CLASS=2
STEP_PIXELS=1024
#XAI_METHODS="gradcamv2 scorecam crp"
XAI_METHODS="gradcamv2 hirescam gradcam++"
#XAI_METHODS="hirescam"

#sbatch run_xai_layermap_bs_testset.sh DME DME_binary_all_split SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-\{0\}-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

#sbatch run_xai_layermap_bs_testset.sh AMD AMD_all_split SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/AMD_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-enc--/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

# ============================================================================
# Run XAI for base models on the full test split (DME_finetuned)
# ============================================================================
python "$SCRIPT" \
        --data_path "$DATA_PATH" \
        --img_dir "$IMG_DIR" \
        --use_split test \
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
