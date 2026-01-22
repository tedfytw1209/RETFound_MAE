#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate octxai
# Go to home directory
#cd $HOME
STUDY=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
RESUME=${4:-"0"} # resume path
Num_CLASS=${5:-"2"} # 2 for AMD, 5 for DR, 5 for Glaucoma, 2 for Cataract
INPUT_SIZE=${6:-"224"}
XAI=${7:-"attn"} # attn, rise, gradcam
STEP_PIXELS=${8:-"224"}
Thickness_DIR=${9:-"/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/"}
SMPMode=${10:-"dec"} # dec, enc, fuse
SMPFuseMode=${11:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${12:-0.5} # 0.0-1.0
SMPSizeMatch=${13:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${14:-0} # 0 for default
ALIGN=${15:-"pre"} # 0 for default
ENC_IDX=${16:-"-1"} # -1 for last encoder layer
DEC_IDX=${17:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${18:-"encoder"} # encoder, decoder, head
SELECT_INDEX=${19:-"-1"} # -1 for last layer
SMPClassifier=${20:-"linear"} # linear, conv
ADDCMD=${21:-""}
ADDCMD2=${22:-""}
ADDCMD3=${23:-""}
NUM_K="0"
data_type="CellData"
Modality="OCT"
IMG_Path="/orange/ruogu.fang/tienyuchang/CellData/"
MASK_DIR="/orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $Num_CLASS


# sbatch finetune_retfound_Celldata_eval.sh DME_all RETFound_mae RETFound_mae_natureOCT output_dir/DME_all-CellData-all-RETFound_mae_natureOCT-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth 2 224 hirescam 224 /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
# sbatch finetune_retfound_Celldata_eval.sh DME_all timm_efficientnet-b4 timm_efficientnet-b4 output_dir/DME_all-CellData-all-timm_efficientnet-b4-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth 2 224 hirescam 224 /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
# sbatch finetune_retfound_Celldata_eval.sh DME_all vit-base-patch16-224 google/vit-base-patch16-224-in21k output_dir/DME_all-CellData-all-google/vit-base-patch16-224-in21k-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth 2 224 hirescam 224 /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv
# sbatch finetune_retfound_Celldata_eval.sh DME_all resnet-50 microsoft/resnet-50 output_dir/DME_all-CellData-all-microsoft/resnet-50-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth 2 224 hirescam 224 /orange/ruogu.fang/tienyuchang/CellData_masks_multiclass_resnet50_new/ enc weighted_sum 0.5 decoder_to_encoder 0 pre -1 -1 encoder -1 conv

# sbatch finetune_retfound_Celldata_eval.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-\{0\}-fea-1-1-0.5-decoder_to_encoder---/checkpoint-best.pth 2 512 hirescam 1024 /orange/ruogu.fang/tienyuchang/CellData_masks/ enc weighted_sum 0.5 decoder_to_encoder 0 -1 -1 encoder -1 conv

XAI_METHODS=("hirescam" "gradcamv2" "gradcam++" "crp")  # List of XAI methods
#XAI_METHODS=("crp")  # List of XAI methods

for XAI in "${XAI_METHODS[@]}"
do
    TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py --batch_size 2     --model $MODEL     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/${data_type}/${Modality}/${STUDY}.csv --thickness_dir $MASK_DIR --task $STUDY-${data_type}-all-$FINETUNED_MODEL-XAI${XAI}-EVAL/ --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 8 --input_size $INPUT_SIZE --num_k $NUM_K --resume $RESUME --xai $XAI --step_pixels $STEP_PIXELS --SMPMode $SMPMode --output_mask --target_module $TARGET_MODULE --select_index $SELECT_INDEX --smp_fuse_mode $SMPFuseMode --smp_alpha $SMPAlpha --smp_size_match $SMPSizeMatch --fusion_dim $FUSION_DIM --enc_idx $ENC_IDX --dec_idx $DEC_IDX --smp_classifier $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3 
done
