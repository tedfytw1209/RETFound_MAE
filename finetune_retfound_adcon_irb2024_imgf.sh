#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"1e-3"}
Epochs=${5:-"100"}
Num_CLASS=${6:-"2"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
ADDCMD=${9:-""}
ADDCMD2=${10:-""}

data_type="IRB2024_imgf"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_adcon_irb2024_imgf.sh ad_control_detect_data RETFound_mae RETFound_mae_natureOCT 5e-4 100 2 default OCT --bal_sampler
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size 16 --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.2 --nb_classes $Num_CLASS --data_path /blue/ruogu.fang/tienyuchang/IRB2024_DL_data_imgf/${STUDY}.csv --task $STUDY-${data_type}-all-$MODEL-${Modality}-${Eval_score}eval-$ADDCMD-$ADDCMD2/ --eval_score $Eval_score --modality $Modality --subset_ratio 4 --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/ --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k 0 $ADDCMD $ADDCMD2
