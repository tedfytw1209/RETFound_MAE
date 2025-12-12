#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

CKPT_ROOT="/orange/ruogu.fang/tienyuchang/RETFound_ckpt"

bash finetune_retfound_benchmark.sh APTOS2019 RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/APTOS2019/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh MESSIDOR2 RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/MESSIDOR2/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh IDRID_data RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/IDRID/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh PAPILA RETFound_mae 3 RETFound_mae_natureCFP $CKPT_ROOT/PAPILA/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh Glaucoma_fundus RETFound_mae 3 RETFound_mae_natureCFP $CKPT_ROOT/Glaucoma/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh JSIEC RETFound_mae 39 RETFound_mae_natureCFP $CKPT_ROOT/JSIEC/checkpoint-best.pth --eval
bash finetune_retfound_benchmark.sh Retina RETFound_mae 4 RETFound_mae_natureCFP $CKPT_ROOT/Retina/checkpoint-best.pth --eval
#sbatch finetune_retfound_benchmark.sh OCTID RETFound_mae 5 RETFound_mae_natureOCT $CKPT_ROOT/OCTID/checkpoint-best.pth --eval