#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# NCex E5 step 2/3: train the five models on the marker-contaminated UF DME data.
# Uses the unchanged main_finetune.py / main_finetune_smp.py with the paper hyperparameters
# (finetune_retfound_UFbenchmark_irb2024v5_full.sh, finetune_retfound_UFbenchmark_irb2024v5_smp_full.sh);
# only --data_path / --img_dir point at the E5 data. The models are not told about the marker.
# EfficientNet-B4 is trained at 224 like the paper run (evaluated at 380, as in the paper XAI eval).
#
#   bash NCex_E5_train.sh <RUN_ID>                   # submit all five models
#   sbatch NCex_E5_train.sh <RUN_ID> <MODEL_TAG>     # one model: SAIL|ResNet-50|RETFound|EffNet-B4|ViT-B16
# Checkpoints: ${RESULTS_ROOT}/NCex/E5/<RUN_ID>/checkpoints/NCex_E5-<MODEL_TAG>/checkpoint-best.pth
# wandb: project RETFound_MAE (hardcoded in the training mains), run name NCex_E5-<MODEL_TAG>
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID=${1:-E5_v3}
MODEL_TAG=$2
RESULTS_ROOT=/orange/ruogu.fang/tienyuchang/RETfound_XAI_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
E5_DIR=${RESULTS_ROOT}/NCex/E5/${RUN_ID}
CKPT_OUT=${E5_DIR}/checkpoints
MODELS=(SAIL ResNet-50 RETFound EffNet-B4 ViT-B16)

if [ -z "${MODEL_TAG}" ]; then
    for MODEL in "${MODELS[@]}"; do
        sbatch --job-name=NCex_E5_train_${MODEL} "$0" ${RUN_ID} ${MODEL}
    done
    exit 0
fi

date;hostname;pwd
module load conda
conda activate retfound_new
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

DATA_FLAGS="--data_path ${E5_DIR}/e5_data.csv --img_dir ${E5_DIR}/data/trainval/ --modality OCT --nb_classes 2 --num_k 0 --eval_score default --visualize_samples --new_subset_num 0"
OUT_FLAGS="--task NCex_E5-${MODEL_TAG}/ --output_dir ${CKPT_OUT} --log_dir ${CKPT_OUT}"

case "${MODEL_TAG}" in
    SAIL)
        torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune_smp.py --savemodel --global_pool \
            --batch_size 16 --world_size 1 --model SMP --epochs 100 --lr 5e-4 --optimizer adamw --layer_decay 0.65 \
            --weight_decay 1e-4 --drop_path 0.0 --finetune ${SEG_CKPT} --num_workers 8 --input_size 512 \
            --SMPMode fuse --smp_fuse_mode weighted_sum --fusion_dim 9 --align pre --smp_alpha 0.5 \
            --smp_size_match decoder_to_encoder --enc_idx -2 --dec_idx -1 --smp_classifier conv --smp_alpha_type scalar \
            --seg_mask --smp_learnable_alpha ${DATA_FLAGS} ${OUT_FLAGS}
        ;;
    ResNet-50|RETFound|EffNet-B4|ViT-B16)
        case "${MODEL_TAG}" in
            ResNet-50) MODEL=resnet-50;             FINETUNED_MODEL=microsoft/resnet-50;              INPUT_SIZE=224 ;;
            RETFound)  MODEL=RETFound_mae;          FINETUNED_MODEL=RETFound_mae_natureOCT;           INPUT_SIZE=224 ;;
            # EffNet-B4 trains at its native 380, the size NCex_E5_eval.sh evaluates it at (E5_v2 trained at 224)
            EffNet-B4) MODEL=timm_efficientnet-b4;  FINETUNED_MODEL=timm_efficientnet-b4;             INPUT_SIZE=380 ;;
            ViT-B16)   MODEL=vit-base-patch16-224;  FINETUNED_MODEL=google/vit-base-patch16-224-in21k; INPUT_SIZE=224 ;;
        esac
        torchrun --nproc_per_node=1 --master_port=$MASTER_PORT main_finetune.py --savemodel --global_pool \
            --batch_size 16 --world_size 1 --model ${MODEL} --epochs 50 --lr 5e-4 --optimizer adamw --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.2 --finetune ${FINETUNED_MODEL} --num_workers 16 --input_size ${INPUT_SIZE} \
            ${DATA_FLAGS} ${OUT_FLAGS}
        ;;
    *) echo "Unknown model ${MODEL_TAG}"; exit 1 ;;
esac
