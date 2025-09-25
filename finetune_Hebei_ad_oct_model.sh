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

# AD-OCT Model Training Script
# Based on research paper specifications for Alzheimer's Disease detection using OCT images

STUDY=$1
MODEL=${2:-"ad_oct_model"}
FEATURE_CHANNELS=${3:-"256"}  # Number of feature channels (default: 256)
NUM_GROUPS=${4:-"3"}          # Number of polarization feature groups (default: 3)
INCLUDE_LOCALIZATION=${5:-"false"}  # Enable localization head (true/false)

# Training hyperparameters based on research paper
BS=18                         # Batch size as specified in paper
LR=${6:-"7e-5"}              # Learning rate: 7e-5 as specified in paper  
WD=${7:-"1e-2"}              # Weight decay: 1e-2 as specified in paper
EPOCHS="100"                  # Number of training epochs
Num_CLASS=${8:-"2"}          # Number of classes (AD vs Control)
SUBSET_RATIO=${9:-"0"}       # Subset ratio for dataset sampling
Eval_score="roc_auc"         # Evaluation metric
Modality="OCT"               # Modality type
OPTIMIZER="adabelief"        # AdaBelief optimizer as specified in paper
TRANSFORM="3"                # AD-OCT specific data augmentation

# Data paths
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_OCT_thickness/Data/"
data_type="IRB2024v5_Wisely_ADCON_DL_data"

# Scheduler parameters
Scheduler_step=20
Scheduler_gamma=0.5

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Training AD-OCT Model"
echo "Study: $STUDY"
echo "Model: $MODEL"
echo "Feature Channels: $FEATURE_CHANNELS"
echo "Number of Groups: $NUM_GROUPS"
echo "Include Localization: $INCLUDE_LOCALIZATION"
echo "Batch Size: $BS"
echo "Learning Rate: $LR"
echo "Weight Decay: $WD"
echo "Optimizer: $OPTIMIZER"
echo "Transform: $TRANSFORM"
echo "Classes: $Num_CLASS"

# Research paper specifications:
# - Sigmoid function as loss function using BCE loss
# - AdaBelief optimizer
# - Batch size: 18
# - Learning rate: 7e-5  
# - Weight decay: 1e-2
# - Data augmentation: horizontal/vertical flips, rotation (-85° to 85°), random cropping
# - Normalization by mean and standard deviation
# - Two Nvidia RTX 3090 GPUs
# - PyTorch 1.10.0

# Construct localization flag
LOCALIZATION_FLAG=""
if [ "$INCLUDE_LOCALIZATION" = "true" ]; then
    LOCALIZATION_FLAG="--include_localization"
fi

# Usage examples:
# sbatch finetune_ad_oct_model.sh ad_control_detect_data ad_oct_model 256 3 false 7e-5 1e-2 2 0
# sbatch finetune_ad_oct_model.sh ad_control_detect_data ad_oct_model 512 3 true 5e-5 1e-2 2 0

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_Chua_Jacqueline.py \
    --savemodel \
    --global_pool \
    --batch_size $BS \
    --world_size 1 \
    --model $MODEL \
    --feature_channels $FEATURE_CHANNELS \
    --num_groups $NUM_GROUPS \
    $LOCALIZATION_FLAG \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WD \
    --nb_classes $Num_CLASS \
    --data_path /blue/ruogu.fang/tienyuchang/${data_type}/${STUDY}.csv \
    --task $STUDY-${data_type}-${MODEL}-feat${FEATURE_CHANNELS}-grp${NUM_GROUPS}-${Modality}-${Eval_score}eval-subset${SUBSET_RATIO} \
    --eval_score $Eval_score \
    --modality $Modality \
    --img_dir $IMG_Path \
    --finetune $MODEL \
    --num_workers 0 \
    --input_size 224 \
    --num_k 0 \
    --optimizer $OPTIMIZER \
    --momentum 0.9 \
    --lr_scheduler step \
    --schedule_step $Scheduler_step \
    --schedule_gamma $Scheduler_gamma \
    --subset_ratio $SUBSET_RATIO \
    --transform $TRANSFORM \
    --use_focal_loss \
    --focal_gamma 2.0 \
    --early_stopping \
    --patience 15 \
    --visualize_samples
