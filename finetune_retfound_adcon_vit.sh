#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=yonghui.wu
#SBATCH --qos=yonghui.wu

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
SUBSTUDY=$2
Num_CLASS=$3
DIVIDE=${4:-"all"}
NUM_K=${5:-"0"}
ADDCMD=${6:-""}
ADDCMD2=${7:-""}

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 

torchrun --nproc_per_node=1 --master_port=48798 main_finetune.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model vit_base_patch16_224     --epochs 100 --blr 2e-5 --layer_decay 0.65     --weight_decay 0.05 --drop_path 0.2     --nb_classes $Num_CLASS     --data_path /blue/ruogu.fang/tienyuchang/OCTAD_old/all_lists_data-ad_control_repath.csv     --task $STUDY-$DIVIDE-$SUBSTUDY-$ADDCMD-$ADDCMD2/ --img_dir /orange/ruogu.fang/tienyuchang/all_imgs_paired/ --finetune google/vit-base-patch16-224 --num_workers 8 --input_size 224 --num_k $NUM_K $ADDCMD $ADDCMD2
