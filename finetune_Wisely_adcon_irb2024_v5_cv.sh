#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# Go to home directory
#cd $HOME
STUDY=$1
data_type=${2:-"IRB2024v5_ADCON_DL_data"}
SPLIT_DIR=${3:-"/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"}
MODEL=${4:-"resnet18_paper"}
FINETUNED_MODEL=$MODEL
Regularization=${5:-"0.01"} # 0.001 to 10 for regularisation loss
#BS=32
BS=256
LR=${6:-"1e-3"} # 0.01 for FC layers, and 0.0001 for other layers
wd=${7:-"0.01"} # 0.01 default
Epochs="100"
Num_CLASS=${8:-"2"}
SUBSET_RATIO=${9:-"0"}
START_FOLD=${10:-0} # Start fold index for resuming CV
ADDCMD=${11:-""} # Additional command line arguments
Eval_score="roc_auc"
Modality="Thickness"
IMG_Path="/blue/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
#SPLIT_DIR="/blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split"
Scheduler_step=10
Scheduler_gamma=0.5
Relative="Wisely"
NFOLDS=10
FOLDS=(0 1 2 3 4 5 6 7 8 9) # CV folds for training

#data_type="IRB2024v5_Wisely_ADCON_DL_data"
#data_type="IRB2024v5_ADCON_DL_data"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#MASTER_PORT=29501

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
# sbatch finetune_Wisely_adcon_irb2024_v5_cv.sh ad_control_detect_data IRB2024v5_ADCON_DL_data /blue/ruogu.fang/tienyuchang/OCTAD_ML_pipeline/psrs_oct/IRB2024_prev_combined_study2_retinaf/split resnet18_paper 0.01 1e-3 0.01 2 4 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    [ "$fold" -lt "$START_FOLD" ] && continue
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py --savemodel --global_pool --batch_size $BS --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --weight_decay $wd --nb_classes $Num_CLASS --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv --task $STUDY-${data_type}-${Relative}-$MODEL-${Modality}-${Eval_score}eval-subset${SUBSET_RATIO} --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --eval_score $Eval_score --modality $Modality --img_dir $IMG_Path --finetune $FINETUNED_MODEL --num_workers 16 --input_size 128 --num_k 0 --optimizer adamw --momentum 0.9 --lr_scheduler step --schedule_step $Scheduler_step --schedule_gamma $Scheduler_gamma --subset_ratio $SUBSET_RATIO --l1_reg $Regularization --l2_reg $Regularization --transform 2 --split_dir $SPLIT_DIR --cv_folds $NFOLDS --cv_fold $fold $ADDCMD
done
