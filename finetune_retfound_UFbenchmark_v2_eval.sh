#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 5, Glaucoma_all_split 5
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
RESUME=${4:-"0"} # 0 for no resume, 1 for resume
Num_CLASS=${5:-"2"} # 2 for AMD, 5 for DR, 5 for Glaucoma, 2 for Cataract
NUM_K=${6:-"0"}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark_v2_eval.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT output_dir/AMD_all_split-all-RETFound_mae--/checkpoint-best.pth 2
TIMM_FUSED_ATTN=0 python main_XAI_evaluation.py --batch_size 16     --model $MODEL     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/new_v2/split/tune5-eval5/${STUDY}.csv     --task $STUDY-v2-all-$MODEL-XAI-EVAL/ --img_dir /orange/ruogu.fang/tienyuchang/all_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K --resume $RESUME
