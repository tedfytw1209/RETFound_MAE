#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate retfound_new
# SMP training on the AD/MCI/control data (IRB2024v5_ADCON_DL_data), the SMP counterpart of
# finetune_retfound_adcon_irb2024_v5.sh. Same arguments as finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh;
# only the data (DATA_DIR, and its folder name in the run name) differ.
STUDY=$1 # ad_control_detect_data 2, ad_mci_detect_data 2, mci_control_detect_data 2, ad_mci_control_detect_data 3
MODEL=${2:-"SMP"}
FINETUNED_MODEL=${3:-"SMP"}
LR=${4:-"1e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"1e-4"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${9:-0} # 0, 500, 1000
SMPMode=${10:-"dec"} # dec, enc, fuse
SMPFuseMode=${11:-"weighted_sum"} # ("weighted_sum", "add", "channel_merge", "channel_multiply", "multiply")
SMPAlpha=${12:-0.5} # 0.0-1.0
SMPSizeMatch=${13:-"decoder_to_encoder"} # decoder_to_encoder, encoder_to_decoder
FUSION_DIM=${14:-0} # 0 for default
ALIGN=${15:-"pre"} # 0 for default
ENC_IDX=${16:-"-1"} # -1 for last encoder layer
DEC_IDX=${17:-"-1"} # -1 for last decoder layer
SMPClassifier=${18:-"linear"} # linear, conv
ALPHA_TYPE=${19:-"scalar"} # scalar, channel, spatial, se, attn
ADDCMD=${20:-""}
ADDCMD2=${21:-""}
ADDCMD3=${22:-""}


NUM_K=0
DATA_DIR=${DATA_DIR:-"/blue/ruogu.fang/tienyuchang/IRB2024v5_ADCON_DL_data_retinaf_diseaseclean/"} # folder holding ${STUDY}.csv (env DATA_DIR)
data_type=$(basename "$DATA_DIR") # used in the run name
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
Epochs=100
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=16

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $STUDY
echo $Num_CLASS

# SMP-fuse (the UF configuration of multiseed_run_all.sh) on ad_control:
# sbatch finetune_retfound_adcon_irb2024_v5_smp.sh ad_control_detect_data SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth 5e-4 2 1e-4 default OCT 0 fuse weighted_sum 0.5 decoder_to_encoder 9 pre -2 -1 conv scalar --seg_mask --smp_learnable_alpha --bal_sampler
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_smp.py --savemodel --global_pool --batch_size $BATCH_SIZE --world_size 1 --model $MODEL --epochs $Epochs --lr $LR --optimizer $OPTIMIZER --layer_decay 0.65 --weight_decay $weight_decay --drop_path 0.0 --nb_classes $Num_CLASS --data_path ${DATA_DIR%/}/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-trsub${SUBSETNUM}-${SMPMode}-smp${SMPFuseMode}-$ALIGN-$FUSION_DIM-fea${ENC_IDX}${DEC_IDX}-${SMPAlpha}-${SMPSizeMatch}-${SMPClassifier}-alpha${ALPHA_TYPE}-$ADDCMD-$ADDCMD2-$ADDCMD3/ --img_dir $IMG_Path --log_dir /orange/ruogu.fang/tienyuchang/RETfound_results --output_dir /orange/ruogu.fang/tienyuchang/RETfound_results --finetune $FINETUNED_MODEL --num_workers 8 --input_size 512 --num_k $NUM_K --eval_score $Eval_score --modality $Modality --visualize_samples --new_subset_num $SUBSETNUM --SMPMode $SMPMode --smp_fuse_mode $SMPFuseMode --fusion_dim $FUSION_DIM --align $ALIGN --smp_alpha $SMPAlpha --smp_size_match $SMPSizeMatch --enc_idx $ENC_IDX --dec_idx $DEC_IDX --smp_classifier $SMPClassifier --smp_alpha_type $ALPHA_TYPE $ADDCMD $ADDCMD2 $ADDCMD3
