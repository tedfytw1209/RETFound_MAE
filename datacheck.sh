#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

# Modify the path to your singularity container 

torchrun --nproc_per_node=1 --master_port=48798 main_datacheck.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model RETFound_mae     --epochs 50 --blr 5e-4 --layer_decay 0.65     --weight_decay 0.05 --drop_path 0.2     --nb_classes 5     --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/OCTID     --task datacheck/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224

torchrun --nproc_per_node=1 --master_port=48798 main_datacheck.py --savemodel --global_pool    --batch_size 16     --world_size 1     --model RETFound_mae     --epochs 100 --blr 2e-5 --layer_decay 0.65     --weight_decay 0.05 --drop_path 0.2     --nb_classes 2     --data_path /blue/ruogu.fang/tienyuchang/OCTAD_old/all_lists_data-ad_control_repath.csv     --task datacheck/ --img_dir /orange/ruogu.fang/tienyuchang/all_imgs_paired/ --finetune RETFound_mae_natureOCT --num_workers 8 --input_size 224
