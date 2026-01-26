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
XAI=${7:-"attn"} # XAI_METHODS=("hirescam" "gradcam++" "gradcamv2" "crp")  # List of XAI methods
STEP_PIXELS=${8:-"224"}
ADD_WORDS=${9:-""} # additional words to specify the finetuned model
ADD_WORDS2=${10:-""} # additional words to specify the finetuned model
NUM_K="0"
data_type="v5"
data_type="IRB2024_v5_all"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark_v5_eval_full.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0---add_mask---train_no_aug/checkpoint-best.pth 2 224 attn
#sbatch finetune_retfound_UFbenchmark_v5_eval_full.sh DME_binary_all_split RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5_all-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 2 224 hirescam
TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py --batch_size 4     --model $MODEL     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune8-eval2/${STUDY}.csv     --task $STUDY-$data_type-$FINETUNED_MODEL-$ADD_WORDS-XAI${XAI}-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size $INPUT_SIZE --num_k $NUM_K --resume $RESUME --xai $XAI --step_pixels $STEP_PIXELS --output_mask $ADD_WORDS $ADD_WORDS2
