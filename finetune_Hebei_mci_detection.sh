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

# DuCAN Model Training Script
# End-to-end framework for MCI detection based on OCT images and fundus photographs
# Based on research paper specifications with dual-modal cross-attention network

STUDY=$1
MODEL=${2:-"ducan"}
FUNDUS_WEIGHT=${3:-"0.7"}      # α weight for fundus loss (paper specifies 0.7)
OCT_WEIGHT=${4:-"0.7"}         # β weight for OCT loss (paper specifies 0.7)
MULTIMODAL_WEIGHT=${5:-"1.0"}  # Weight for fusion loss (normalized to 1.0)

# Training hyperparameters based on research paper specifications
BS=8                           # Batch size: 8 as specified in paper
LR=${6:-"3e-4"}               # Initial learning rate: 0.0003 as specified in paper
WD=${7:-"1e-2"}               # Weight decay: 0.01 as specified in paper (initial decay factor)
EPOCHS="400"                   # Number of training epochs: 400 as specified in paper
Num_CLASS=${8:-"3"}           # Number of classes (AD, MCI, CN)
SUBSET_RATIO=${9:-"0"}        # Subset ratio for dataset sampling
Eval_score="accuracy"         # Evaluation metric
Modality="dual"               # Dual modality (fundus + OCT)
OPTIMIZER="sgd"               # Stochastic Gradient Descent as specified in paper
TRANSFORM="3"                 # Data augmentation with random cropping and reversing

# Data paths for dual-modal training
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
data_type="IRB2024v5_ADCON_DL_data"

# Scheduler parameters - not specified in paper, using reasonable defaults
Scheduler_step=50
Scheduler_gamma=0.5

# Early stopping parameters based on paper specifications
PATIENCE=50                    # Stop if validation doesn't improve within 50 epochs
EARLY_STOPPING="true"

# Loss function weights based on paper equation (13)
# Lfinal = αLOCT + βLfundus + Lfusion where α=β=0.7
FUNDUS_LOSS_WEIGHT=$FUNDUS_WEIGHT
OCT_LOSS_WEIGHT=$OCT_WEIGHT
MULTIMODAL_LOSS_WEIGHT=$MULTIMODAL_WEIGHT

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

#sbatch finetune_Hebei_mci_detection.sh ad_mci_control_detect_data
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
    --savemodel \
    --global_pool \
    --batch_size $BS \
    --world_size 1 \
    --model $MODEL \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WD \
    --nb_classes $Num_CLASS \
    --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv \
    --task $STUDY-${data_type}-${MODEL}-fundus${FUNDUS_WEIGHT}-oct${OCT_WEIGHT}-multi${MULTIMODAL_WEIGHT}-${Eval_score}eval-subset${SUBSET_RATIO} \
    --eval_score $Eval_score \
    --modality $Modality \
    --img_dir $IMG_Path \
    --finetune $MODEL \
    --num_workers 4 \
    --input_size 224 \
    --num_k 0 \
    --optimizer $OPTIMIZER \
    --momentum 0.9 \
    --lr_scheduler step \
    --schedule_step $Scheduler_step \
    --schedule_gamma $Scheduler_gamma \
    --subset_ratio $SUBSET_RATIO \
    --transform $TRANSFORM \
    --fundus_loss_weight $FUNDUS_LOSS_WEIGHT \
    --oct_loss_weight $OCT_LOSS_WEIGHT \
    --multimodal_loss_weight $MULTIMODAL_LOSS_WEIGHT \
    --early_stopping \
    --patience $PATIENCE \
    --visualize_samples \
    --use_ducan_preprocessing \
    --warmup_epochs 10 \
    --min_lr 1e-6 \
    --clip_grad 1.0

