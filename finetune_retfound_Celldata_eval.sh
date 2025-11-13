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
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
RESUME=${4:-"0"} # resume path
Num_CLASS=${5:-"2"} # 2 for AMD, 5 for DR, 5 for Glaucoma, 2 for Cataract
INPUT_SIZE=${6:-"224"}
XAI=${7:-"attn"} # attn, rise, gradcam
STEP_PIXELS=${8:-"224"}
NUM_K="0"
data_type="CellData"
Modality="OCT"
IMG_Path="/orange/ruogu.fang/tienyuchang/CellData/"
MASK_DIR="/orange/ruogu.fang/tienyuchang/CellData_masks/"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_Celldata_eval.sh DME_all RETFound_mae RETFound_mae_natureOCT output_dir/DME_all-CellData-all-RETFound_mae_natureOCT-OCT-bs16ep3lr5e-4optadamw-defaulteval--/checkpoint-best.pth 2 224 attn
TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py --batch_size 4     --model $MODEL     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/${data_type}/${Modality}/${STUDY}.csv     --task $STUDY-${data_type}-all-$FINETUNED_MODEL-XAI${XAI}-EVAL/ --img_dir $IMG_Path --thickness_dir $MASK_DIR --finetune $FINETUNED_MODEL --num_workers 8 --input_size $INPUT_SIZE --num_k $NUM_K --resume $RESUME --xai $XAI --step_pixels $STEP_PIXELS --output_mask
