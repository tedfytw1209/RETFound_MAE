#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

# ─────────────────────────────────────────────────────────────────────────────
# NCex E5 step 3/3: evaluate the E5-trained models on T1 (clean), T2 (all marked), T3 (healthy marked).
# cheat_reliance = AUROC(T1) - AUROC(T3); on T2: marker_mass and pointing_hit for Grad-CAM, Grad-CAM++, HiResCAM.
#
#   bash NCex_E5_eval.sh <RUN_ID>                 # five model jobs -> merge (afterok)
#   sbatch NCex_E5_eval.sh <RUN_ID> <MODEL_TAG>   # one model: SAIL|ResNet-50|RETFound|EffNet-B4|ViT-B16
#   sbatch NCex_E5_eval.sh <RUN_ID> merge
# Results: ${RESULTS_ROOT}/NCex/E5/<RUN_ID>/{shortcut.csv,config.yaml,examples/,parts/,configs/}
# ─────────────────────────────────────────────────────────────────────────────

RUN_ID=${1:-E5_v1}
MODEL_TAG=$2
RESULTS_ROOT=/orange/ruogu.fang/tienyuchang/RETfound_XAI_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
CKPT_OUT=${RESULTS_ROOT}/NCex/E5/${RUN_ID}/checkpoints
MODELS=(SAIL ResNet-50 RETFound EffNet-B4 ViT-B16)
COMMON="--run_id ${RUN_ID} --results_root ${RESULTS_ROOT} --modality OCT --nb_classes 2 --disease_class 1 --batch_size 8 --num_workers 8"

if [ -z "${MODEL_TAG}" ]; then
    JOBS=""
    for MODEL in "${MODELS[@]}"; do
        JID=$(sbatch --parsable --job-name=NCex_E5_eval_${MODEL} "$0" ${RUN_ID} ${MODEL})
        JOBS="${JOBS}:${JID}"
    done
    sbatch --job-name=NCex_E5_merge --dependency=afterok${JOBS} "$0" ${RUN_ID} merge
    exit 0
fi

date;hostname;pwd
module load conda
conda activate octxai

RESUME="--resume ${CKPT_OUT}/NCex_E5-${MODEL_TAG}/checkpoint-best.pth"
case "${MODEL_TAG}" in
    SAIL)      FLAGS="--model SMP --finetune ${SEG_CKPT} --input_size 512 --SMPMode fuse --smp_fuse_mode weighted_sum --smp_alpha 0.5 --smp_size_match decoder_to_encoder --fusion_dim 9 --align pre --enc_idx -2 --dec_idx -1 --target_module head --select_index -1 --smp_classifier conv --seg_mask --smp_learnable_alpha ${RESUME}" ;;
    ResNet-50) FLAGS="--model resnet-50 --finetune microsoft/resnet-50 --input_size 224 ${RESUME}" ;;
    RETFound)  FLAGS="--model RETFound_mae --finetune RETFound_mae_natureOCT --input_size 224 ${RESUME}" ;;
    EffNet-B4) FLAGS="--model timm_efficientnet-b4 --finetune timm_efficientnet-b4 --input_size 380 ${RESUME}" ;;
    ViT-B16)   FLAGS="--model vit-base-patch16-224 --finetune google/vit-base-patch16-224-in21k --input_size 224 ${RESUME}" ;;
    merge)
        python NCex_E5_eval.py --mode merge ${COMMON}
        exit 0 ;;
    *) echo "Unknown model ${MODEL_TAG}"; exit 1 ;;
esac

TIMM_FUSED_ATTN=0 python NCex_E5_eval.py --mode run --model_tag ${MODEL_TAG} ${FLAGS} ${COMMON}
