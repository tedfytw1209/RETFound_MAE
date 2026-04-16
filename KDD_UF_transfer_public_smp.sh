#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

BASE_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
RESULTS_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
DATASET=DME_binary_all_split
DATA_TYPE=IRB2024_v5_all
MODALITY=OCT
LR=5e-4
BATCH_SIZE=16
EPOCHS=100
FIXED_DEC_IDX=-1   # fixed DEC_IDX (pos 17 in training script, always -1)
ALPHA_TYPE=scalar
NUM_CLASS=2
INPUT_SIZE=512
STEP_PIXELS=1024
FUSION_DIM=9

# Public dataset study names (equivalent disease split used in transfer eval)
OCTDL_STUDY=DME_all
CELLDATA_STUDY=DME_all

SEG_PATH=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
Thickness_DIR=/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/
RESUME=/orange/ruogu.fang/tienyuchang/RETfound_results/${DATASET}-IRB2024_v5_all-all-${SEG_PATH}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-/checkpoint-best.pth

# --- Eval on OCTDL ---
XAI_METHODS=("hirescam" "gradcamv2" "gradcam++")  # List of XAI methods

for XAI in "${XAI_METHODS[@]}"
do
    bash finetune_retfound_OCTDL_eval.sh ${OCTDL_STUDY} SMP ${BASE_CKPT} "${RESUME}" ${NUM_CLASS} ${INPUT_SIZE} ${XAI} ${STEP_PIXELS} /orange/ruogu.fang/tienyuchang/OCTDL_masks_multiclass_resnet50_new/ fuse weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre -2 ${FIXED_DEC_IDX} head -1 conv "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"

    # --- Eval on CellData ---
    bash finetune_retfound_Celldata_eval.sh ${CELLDATA_STUDY} SMP ${BASE_CKPT} "${RESUME}" ${NUM_CLASS} ${INPUT_SIZE} ${XAI} ${STEP_PIXELS} /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ fuse weighted_sum 0.5 decoder_to_encoder ${FUSION_DIM} pre -2 ${FIXED_DEC_IDX} head -1 conv "--seg_mask" "--smp_learnable_alpha" "--smp_alpha_type ${ALPHA_TYPE}"
done