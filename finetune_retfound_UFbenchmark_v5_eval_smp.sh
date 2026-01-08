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

date;hostname;pwd

module load conda
conda activate octxai
# Go to home directory
#cd $HOME
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
RESUME=${4:-"0"} # resume path
Num_CLASS=${5:-"2"} # 2 for AMD, 5 for DR, 5 for Glaucoma, 2 for Cataract
INPUT_SIZE=${6:-"224"}
XAI=${7:-"attn"} # attn, rise, gradcam
STEP_PIXELS=${8:-"224"}
Thickness_DIR=${9:-"/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"}
SMPMode=${10:-"dec"} # dec, enc, fuse
SMPFuseMode=${11:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${12:-0.5} # 0.0-1.0
SMPSizeMatch=${13:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${14:-0} # 0 for default
ENC_IDX=${15:-"-1"} # -1 for last encoder layer
DEC_IDX=${16:-"-1"} # -1 for last decoder layer
TARGET_MODULE=${17:-"encoder"} # encoder, decoder, head
SELECT_INDEX=${18:-"-1"} # -1 for last layer
SMPClassifier=${19:-"linear"} # linear, conv
ADDCMD=${20:-""}
ADDCMD2=${21:-""}
ADDCMD3=${22:-""}
NUM_K="0"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

#sbatch finetune_retfound_UFbenchmark_v5_eval_smp.sh DME_binary_all_split SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs4ep20lr1e-4optadamw-defaulteval-trsub0-dec---add_mask---train_no_aug/checkpoint-best.pth 2 512 gradcam++ 1024 /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ dec weighted_sum 0.5 decoder_to_encoder -1 -1 decoder -1 --add_mask
#sbatch finetune_retfound_UFbenchmark_v5_eval_smp.sh DME_binary_all_split SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth-OCT-bs8ep50lr1e-4optadamw-defaulteval-trsub0-enc-smpweighted_sum-\{0\}-fea-1-1-0.5-decoder_to_encoder---/checkpoint-best.pth 2 512 hirescam 1024 /orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/ enc weighted_sum 0.5 decoder_to_encoder 0 -1 -1 encoder -1 linear

TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py --batch_size 2     --model $MODEL     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5/split/tune5-eval5/${STUDY}.csv     --task $STUDY-v5-all-$FINETUNED_MODEL-$ADD_WORDS-XAI${XAI}-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size $INPUT_SIZE --num_k $NUM_K --resume $RESUME --xai $XAI --step_pixels $STEP_PIXELS --SMPMode $SMPMode --output_mask $ADD_WORDS --thickness_dir $Thickness_DIR --target_module $TARGET_MODULE --select_index $SELECT_INDEX --smp_fuse_mode $SMPFuseMode --smp_alpha $SMPAlpha --smp_size_match $SMPSizeMatch --fusion_dim $FUSION_DIM --enc_idx $ENC_IDX --dec_idx $DEC_IDX --smp_classifier $SMPClassifier $ADDCMD $ADDCMD2 $ADDCMD3 
