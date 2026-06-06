#!/bin/bash
#SBATCH --job-name=transfer_imgs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16gb
#SBATCH --time=144:00:00
#SBATCH --output=transfer_imgs.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

SRC="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
DST="/blue/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"

mkdir -p "$DST"

# Use GNU parallel to run 32 concurrent rsync streams (one per top-level subdir)
# Falls back to single rsync if no subdirs exist
SUBDIRS=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d)

if [ -z "$SUBDIRS" ]; then
    echo "No subdirectories found, running single rsync..."
    rsync -av --no-perms --info=progress2 "$SRC" "$DST"
else
    echo "Running parallel rsync with $SLURM_CPUS_PER_TASK streams..."
    echo "$SUBDIRS" | xargs -P "$SLURM_CPUS_PER_TASK" -I{} \
        rsync -a --no-perms --info=progress2 {} "$DST"/
fi

echo "Transfer done. Verifying file count..."
echo "Source: $(find "$SRC" -type f | wc -l) files"
echo "Dest:   $(find "$DST" -type f | wc -l) files"
