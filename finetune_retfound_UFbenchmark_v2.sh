#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_all_split_binary 2, Glaucoma_all_split_binary 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
NUM_K=${7:-"0"}
ADDCMD=${8:-""}
ADDCMD2=${9:-""}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_UFbenchmark_v2.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT 5e-4 2
# sbatch finetune_retfound_UFbenchmark_v2.sh AMD_all_split RETFound_dinov2 RETFound_dinov2_meh 5e-4 2
torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model $MODEL     --epochs 100 --lr $LR --layer_decay 0.65     --weight_decay $weight_decay --drop_path 0.2     --nb_classes $Num_CLASS     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/new_v2/split/tune5-eval5/${STUDY}.csv     --task $STUDY-all-$MODEL-$ADDCMD-$ADDCMD2/ --img_dir /orange/ruogu.fang/tienyuchang/all_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k $NUM_K $ADDCMD $ADDCMD2
