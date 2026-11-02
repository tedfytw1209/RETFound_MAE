#!/bin/bash
#SBATCH --job-name=octxai_infer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang
#
# RETFound / SMP-fuse inference wrapper (probabilities + saliency maps). Works locally and via sbatch.
#
# Usage:
#   bash   run_inference.sh <image|dir|csv> [more inputs...]
#   sbatch run_inference.sh <image|dir|csv> [more inputs...]
#
# Configure with environment variables (defaults in brackets):
#   MODEL         weights name / dir / training checkpoint   (required)
#   OUTPUT        output CSV                                 [predictions.csv]
#   XAI           hirescam | gradcamv2 | gradcam++           [from the model card]
#   XAI_TARGET    pred | 0 | 1 ...  class to explain         [pred]
#   NO_SALIENCY   1 = probabilities only                     [0]
#   BATCH_SIZE    images per forward pass                    [8]
#   IMAGE_ROOT    root for relative paths in input CSVs
#   CLASS_NAMES   e.g. Control,DME (raw training checkpoints only)
#   EXTRA_ARGS    anything else passed straight to predict.py
#
# Examples:
#   MODEL=smp_fuse_DME bash run_inference.sh scan.jpg
#   MODEL=retfound_DME OUTPUT=out/retfound.csv sbatch run_inference.sh bscans/
#   MODEL=smp_fuse_DME XAI=gradcamv2 XAI_TARGET=1 IMAGE_ROOT=/orange/.../IRB2024_imgs_paired/ \
#     OUTPUT=out/smp.csv sbatch run_inference.sh list.csv

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -lt 1 ] || [ -z "${MODEL:-}" ]; then
    sed -n '12,32p' "$0"
    exit 1
fi

# Settings must come BEFORE the command (MODEL=x bash run_inference.sh in.jpg), not after it.
for a in "$@"; do
    if [[ "$a" =~ ^[A-Z_]+= ]] || [ "$a" = "bash" ] || [ "$a" = "sbatch" ]; then
        echo "ERROR: '$a' was passed as an input." >&2
        echo "  Put settings before the command, e.g.: MODEL=smp_fuse_DME OUTPUT=out/p.csv bash run_inference.sh scan.jpg" >&2
        exit 1
    fi
done

# On HiPerGator, use the environment the models were trained in
if command -v module >/dev/null 2>&1; then
    module load conda
    conda activate octxai
fi

ARGS=(--input "$@" --model "$MODEL" --output "${OUTPUT:-predictions.csv}" --batch_size "${BATCH_SIZE:-8}")
[ -n "${XAI:-}" ]         && ARGS+=(--xai "$XAI")
[ -n "${XAI_TARGET:-}" ]  && ARGS+=(--xai_target "$XAI_TARGET")
[ "${NO_SALIENCY:-0}" = "1" ] && ARGS+=(--no_saliency)
[ -n "${IMAGE_ROOT:-}" ]  && ARGS+=(--image_root "$IMAGE_ROOT")
[ -n "${CLASS_NAMES:-}" ] && ARGS+=(--class_names "$CLASS_NAMES")
[ -n "${SLURM_CPUS_PER_TASK:-}" ] && ARGS+=(--n_workers "$SLURM_CPUS_PER_TASK")

# shellcheck disable=SC2086
python predict.py "${ARGS[@]}" ${EXTRA_ARGS:-}
