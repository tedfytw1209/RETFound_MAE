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
# NCex E4: saliency sanity checks (Test A cascading re-init, Test B disease vs healthy logit).
# Models: SAIL, ResNet-50, RETFound, EfficientNet-B4, ViT-B/16 at their native input sizes.
# Data:   100 test scans (50/50) from UF DME and from OCTDL ERM, the same scans for every model.
#
# Launch everything (samples -> 10 model jobs -> merge, chained with afterok):
#   bash NCex_E4_sanity.sh                     # RUN_ID defaults to E4_<date>
#   RUN_ID=E4_debug N_PER_CLASS=5 bash NCex_E4_sanity.sh
# Pre-flight (login node, no job): bash NCex_E4_sanity.sh check
# Single stages (what the launcher submits):
#   sbatch NCex_E4_sanity.sh samples <UF_DME|OCTDL_ERM> <RUN_ID>
#   sbatch NCex_E4_sanity.sh run <SAIL|ResNet-50|RETFound|EffNet-B4|ViT-B16> <UF_DME|OCTDL_ERM> <RUN_ID>
#   sbatch NCex_E4_sanity.sh merge <RUN_ID>
# Results: ${RESULTS_ROOT}/NCex/E4/<RUN_ID>/{sanity.csv,config.yaml,examples/,parts/,configs/}
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_ROOT=/orange/ruogu.fang/tienyuchang/RETfound_XAI_results
CKPT_DIR=/orange/ruogu.fang/tienyuchang/RETfound_results
SEG_CKPT=/blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass_resnet50_new.pth
N_PER_CLASS=${N_PER_CLASS:-50}
MODELS=(SAIL ResNet-50 RETFound EffNet-B4 ViT-B16)
DATASETS=(UF_DME OCTDL_ERM)

SAIL_FLAGS="--model SMP --finetune ${SEG_CKPT} --input_size 512 --SMPMode fuse --smp_fuse_mode weighted_sum --smp_alpha 0.5 --smp_size_match decoder_to_encoder --fusion_dim 9 --align pre --enc_idx -2 --dec_idx -1 --target_module head --select_index -1 --smp_classifier conv --seg_mask --smp_learnable_alpha"
SAIL_SUFFIX="fuse-smpweighted_sum-pre-9-fea-2-1-0.5-decoder_to_encoder-conv---seg_mask---smp_learnable_alpha-"

dataset_flags() {
    # $1=dataset tag -> data flags
    case "$1" in
        UF_DME)    echo "--data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/IRB2024_v5_all/split/tune8-eval2/DME_binary_all_split.csv --img_dir /orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/" ;;
        OCTDL_ERM) echo "--data_path /orange/ruogu.fang/tienyuchang/OCTDL/ERM_all.csv --img_dir /orange/ruogu.fang/tienyuchang/OCTDL/" ;;
        *) echo "Unknown dataset $1" >&2; exit 1 ;;
    esac
}

baseline_resume() {
    # $1=dataset tag $2=FINETUNED_MODEL -> paper checkpoint (same paths as baseline_multirun_XAI_eval_full.sh / KDD_public_transfer_UF_baselines.sh)
    case "$1" in
        UF_DME)    echo "${CKPT_DIR}/DME_binary_all_split-IRB2024_v5_all-all-$2-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth" ;;
        OCTDL_ERM) echo "${CKPT_DIR}/ERM_all-OCTDL-all-$2-OCT-bs16ep50lr5e-4optadamw-defaulteval--/checkpoint-best.pth" ;;
    esac
}

model_flags() {
    # $1=model tag $2=dataset tag -> model flags incl. --resume
    case "$1" in
        SAIL)
            case "$2" in
                UF_DME)    RESUME="${CKPT_DIR}/DME_binary_all_split-IRB2024_v5_all-all-${SEG_CKPT}-OCT-bs16ep100lr5e-4optadamw-defaulteval-trsub0-${SAIL_SUFFIX}/checkpoint-best.pth" ;;
                OCTDL_ERM) RESUME="${CKPT_DIR}/ERM_all-OCTDL-all-${SEG_CKPT}-OCT-bs16ep50lr1e-4optadamw-defaulteval-trsub0-${SAIL_SUFFIX}/checkpoint-best.pth" ;;
            esac
            echo "${SAIL_FLAGS} --resume ${RESUME}" ;;
        ResNet-50) echo "--model resnet-50 --finetune microsoft/resnet-50 --input_size 224 --resume $(baseline_resume $2 microsoft/resnet-50)" ;;
        RETFound)  echo "--model RETFound_mae --finetune RETFound_mae_natureOCT --input_size 224 --resume $(baseline_resume $2 RETFound_mae_natureOCT)" ;;
        EffNet-B4) echo "--model timm_efficientnet-b4 --finetune timm_efficientnet-b4 --input_size 380 --resume $(baseline_resume $2 timm_efficientnet-b4)" ;;
        ViT-B16)   echo "--model vit-base-patch16-224 --finetune google/vit-base-patch16-224-in21k --input_size 224 --resume $(baseline_resume $2 google/vit-base-patch16-224-in21k)" ;;
        *) echo "Unknown model $1" >&2; exit 1 ;;
    esac
}

COMMON="--nb_classes 2 --batch_size 8 --num_workers 8 --results_root ${RESULTS_ROOT} --n_per_class ${N_PER_CLASS}"

flag_value() {
    # $1=flag string $2=flag name -> value following the flag
    echo "$1" | awk -v k="$2" '{for (i = 1; i < NF; i++) if ($i == k) print $(i + 1)}'
}

STAGE=$1
if [ "${STAGE}" == "check" ]; then
    # ── pre-flight: every data CSV, image dir and checkpoint exists (no job submitted) ──
    N_MISSING=0
    for DATASET in "${DATASETS[@]}"; do
        DFLAGS=$(dataset_flags ${DATASET})
        for P in "$(flag_value "${DFLAGS}" --data_path)" "$(flag_value "${DFLAGS}" --img_dir)"; do
            if [ -e "${P}" ]; then echo "OK      ${DATASET} data ${P}"; else echo "MISSING ${DATASET} data ${P}"; N_MISSING=$((N_MISSING + 1)); fi
        done
        for MODEL in "${MODELS[@]}"; do
            P=$(flag_value "$(model_flags ${MODEL} ${DATASET})" --resume)
            if [ -f "${P}" ]; then echo "OK      ${MODEL}/${DATASET} ${P}"; else echo "MISSING ${MODEL}/${DATASET} ${P}"; N_MISSING=$((N_MISSING + 1)); fi
        done
    done
    if [ -f "${SEG_CKPT}" ]; then echo "OK      Stage-I ${SEG_CKPT}"; else echo "MISSING Stage-I ${SEG_CKPT}"; N_MISSING=$((N_MISSING + 1)); fi
    echo "${N_MISSING} missing"
    exit ${N_MISSING}
fi
if [ -z "${STAGE}" ]; then
    # ── launcher ──
    RUN_ID=${RUN_ID:-E4_$(date +%Y%m%d_%H%M)}
    echo "Submitting E4 run ${RUN_ID}"
    SAMPLE_JOBS=""
    for DATASET in "${DATASETS[@]}"; do
        JID=$(sbatch --parsable --job-name=NCex_E4_samples "$0" samples ${DATASET} ${RUN_ID})
        SAMPLE_JOBS="${SAMPLE_JOBS}:${JID}"
    done
    RUN_JOBS=""
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            JID=$(sbatch --parsable --job-name=NCex_E4_${MODEL}_${DATASET} --dependency=afterok${SAMPLE_JOBS} "$0" run ${MODEL} ${DATASET} ${RUN_ID})
            RUN_JOBS="${RUN_JOBS}:${JID}"
        done
    done
    sbatch --job-name=NCex_E4_merge --dependency=afterok${RUN_JOBS} "$0" merge ${RUN_ID}
    exit 0
fi

date;hostname;pwd
module load conda
conda activate octxai

case "${STAGE}" in
    samples)
        DATASET=$2; RUN_ID=$3
        python NCex_E4_sanity.py --mode samples --run_id ${RUN_ID} --dataset_tag ${DATASET} $(dataset_flags ${DATASET}) ${COMMON} --input_size 224
        ;;
    run)
        MODEL=$2; DATASET=$3; RUN_ID=$4
        TIMM_FUSED_ATTN=0 python NCex_E4_sanity.py --mode run --run_id ${RUN_ID} --model_tag ${MODEL} --dataset_tag ${DATASET} $(dataset_flags ${DATASET}) $(model_flags ${MODEL} ${DATASET}) ${COMMON}
        ;;
    merge)
        RUN_ID=$2
        python NCex_E4_sanity.py --mode merge --run_id ${RUN_ID} ${COMMON}
        ;;
    *) echo "Unknown stage ${STAGE}"; exit 1 ;;
esac
