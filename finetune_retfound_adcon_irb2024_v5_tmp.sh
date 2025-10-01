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
LAYER_IDX=$2 # 0, 1, 2, 3
MODEL="RETFound_mae"
FINETUNED_MODEL="RETFound_mae_natureCFP"
LR="5e-4"
Epochs=50
Num_CLASS=2
Eval_score="default"
Modality="Thickness" # CFP, OCT, OCT_CFP
SUBSETNUM=0 # 0, 500, 1000
SUBSET_RATIO=0
ADDCMD="--bal_sampler"
ADDCMD2="--th_heatmap"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"

data_type="IRB2024v5_ADCON_DL_data"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_retfound_adcon_irb2024_v5_tmp.sh ad_control_detect_data 0
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool --batch_size 16 --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.2 --nb_classes $Num_CLASS --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv --task $STUDY-${data_type}-all-$MODEL-${Modality}_${LAYER_IDX}-${Eval_score}eval-sub$SUBSET_RATIO-$ADDCMD-$ADDCMD2/ --eval_score $Eval_score --modality $Modality --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 8 --input_size 224 --num_k 0 --subset_ratio $SUBSET_RATIO --select_layer_idx $LAYER_IDX $ADDCMD $ADDCMD2
