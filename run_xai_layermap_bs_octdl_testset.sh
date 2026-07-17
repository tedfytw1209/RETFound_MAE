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
# Full-test-set variant of run_xai_layermap_bs_octdl.sh: instead of the hand-picked
# "<task>_sampled" folder, this passes --data_path/--img_dir/--use_split test so
# case_study_SMP_layermap_bs.py loads the full test split via util.datasets.build_dataset()
# (the same function main_XAI_evaluation.py uses) -- see its _load_full_split_data().
# Masks/layer-lines aren't meaningful for OCTDL's segmentation masks, so this keeps
# --no_load_mask --no_draw_layer, same as the existing convention.

SCRIPT="case_study_SMP_layermap_bs.py"
STUDY=${1:-"OCTDL_DME"} # task label, used for --task / output dir naming (kept independent of the CSV filename)
STUDY_NAME=${2:-"DME_all"} # full test-split CSV base filename (no .csv) under OCTDL, e.g. DME_all, AMD_all
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

OUTPUT_DIR="./heatmap_params_octdl_dme_full"

# ============================================================================
# Common parameters (full OCTDL test split)
# ============================================================================
DATA_TYPE="OCTDL"
DATA_PATH="/orange/ruogu.fang/tienyuchang/${DATA_TYPE}/${STUDY_NAME}.csv"
IMG_DIR="/orange/ruogu.fang/tienyuchang/OCTDL/"
THICKNESS_DIR="/orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/"
BATCH_SIZE=8
NUM_SAMPLES=-1 # -1 = every image in the test split (set >0 to still sub-sample for a quick look)
Num_CLASS=2
STEP_PIXELS=1024
#XAI_METHODS="gradcamv2 scorecam crp"
XAI_METHODS="gradcamv2 hirescam gradcam++"
#XAI_METHODS="hirescam"

#sbatch run_xai_layermap_bs_octdl_testset.sh OCTDL_DME DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-\{0\}-fea-1-1-0.5-decoder_to_encoder-conv---/checkpoint-best.pth 512 enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

#sbatch run_xai_layermap_bs_octdl_testset.sh OCTDL_DME DME_all RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/DME_all-OCTDL-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
#sbatch run_xai_layermap_bs_octdl_testset.sh OCTDL_DME DME_all vit-base-patch16-224 google/vit-base-patch16-224-in21k /orange/ruogu.fang/tienyuchang/RETfound_results/DME_all-OCTDL-all-google/vit-base-patch16-224-in21k-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth 224 dec weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

#sbatch run_xai_layermap_bs_octdl_testset.sh OCTDL_DME DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpmultiply-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask--/checkpoint-best.pth 512 fuse multiply 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask
#sbatch run_xai_layermap_bs_octdl_testset.sh OCTDL_DME DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_all-OCTDL-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth 512 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 head -1 conv --seg_mask --smp_learnable_alpha

## Related work using run_xai_layermap_multirun_octdl_testset.sh

# ============================================================================
# Run XAI for base models on the full test split (DME_finetuned)
# ============================================================================
python "$SCRIPT" \
        --data_path "$DATA_PATH" \
        --img_dir "$IMG_DIR" \
        --use_split test \
        --thickness_dir "$THICKNESS_DIR" \
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
        --no_load_mask \
        --no_draw_layer \
        --output_dir "$OUTPUT_DIR" \
        $ADDCMD $ADDCMD2 $ADDCMD3
