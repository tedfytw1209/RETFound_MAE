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
FINETUNED_MODEL=$MODEL
BS=${3:-"32"} # 16,32,64
LR=${4:-"1e-3"} # 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05
wd=${5:-"0.005"} # 0.005, 0.0005
Epochs="50"
Num_CLASS=${6:-"2"}
Eval_score="roc_auc"
Modality="Thickness"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
Patience="5"

data_type="Chua_Jacqueline_IRB2024_v5"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_relative_adcon_irb2024_v5.sh ad_control_detect_data alexnet 16 1e-4 0.005 2
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --global_pool --batch_size $BS --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --weight_decay $wd --nb_classes $Num_CLASS --data_path /blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data/${STUDY}.csv --task $STUDY-${data_type}-all-$MODEL-${Modality}-${Eval_score}eval/ --eval_score $Eval_score --modality $Modality --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 0 --input_size 224 --num_k 0 --optimizer sgd --momentum 0.9 --lr_scheduler false --early_stopping --patience $Patience
