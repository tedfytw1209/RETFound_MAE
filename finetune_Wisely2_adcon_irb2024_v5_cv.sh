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
MODEL=${2:-"dual_input_cnn"}
INPUT_MODE=${3:-"images_only"} # all, images_only, gc_ipl_only, octa_only, quantitative_only, gc_ipl_quantitative, octa_quantitative
Regularization=${4:-"0.01"} # 0.001 to 10 for regularisation loss
BS=128 #32*4=128
LR=${5:-"1e-4"} # Adam optimizer learning rate (paper uses adaptive moment estimation)
wd=${6:-"0.01"} # Weight decay 0.01 as specified in paper
#Epochs="100"
Epochs="50"
Num_CLASS=${7:-"2"}
SUBSET_RATIO=${8:-"1.3"}
Eval_score="roc_auc" # AUC as primary performance metric
Modality="Thickness"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
Scheduler_step=10
Scheduler_gamma=0.5
Quantitative_Features=${9:-"10"} # Number of quantitative features
ADDCMD=${10:-""} # Additional command line arguments
Relative="Wisely2"
NFOLDS=5
FOLDS=(0 1 2 3 4)

data_type="IRB2024v5_Wisely_ADCON_DL_data"

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#MASTER_PORT=29501

echo $SUBSTUDY
echo $Num_CLASS

# Paper-specified hyperparameters:
# - 100 epochs
# - Adam optimizer (adaptive moment estimation)
# - Weight decay 0.01
# - Binary cross entropy loss
# - AUC as evaluation metric
# - Youden index optimization for thresholding

# Usage examples:
# sbatch finetune_Wisely2_adcon_irb2024_v5.sh mci_control_detect_data dual_input_cnn images_only 0.01 1e-4 0.01 2 1.3 10 --use_img_per_patient
for fold in ${FOLDS[@]}; do
    torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
        --savemodel \
        --global_pool \
        --batch_size $BS \
        --world_size 1 \
        --model $MODEL \
        --input_mode $INPUT_MODE \
        --quantitative_features $Quantitative_Features \
        --epochs $Epochs \
        --lr $LR \
        --weight_decay $wd \
        --nb_classes $Num_CLASS \
        --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv \
        --task $STUDY-${data_type}-${Relative}-$MODEL-${INPUT_MODE}-${Modality}-${Eval_score}eval-subset${SUBSET_RATIO} \
        --eval_score $Eval_score \
        --modality $Modality \
        --img_dir $IMG_Path \
        --finetune $MODEL \
        --num_workers 0 \
        --input_size 224 \
        --num_k 0 \
        --optimizer adam \
        --momentum 0.9 \
        --lr_scheduler step \
        --schedule_step $Scheduler_step \
        --schedule_gamma $Scheduler_gamma \
        --subset_ratio $SUBSET_RATIO \
        --l1_reg $Regularization \
        --l2_reg $Regularization \
        --transform 2 \
        --cv_folds $NFOLDS --cv_fold $fold \
        $ADDCMD
done