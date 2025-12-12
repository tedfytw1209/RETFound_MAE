#!/bin/bash

CKPT_ROOT="/orange/ruogu.fang/tienyuchang/RETFound_ckpt"

sbatch finetune_retfound_benchmark.sh APTOS2019 RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/APTOS2019/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh MESSIDOR2 RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/MESSIDOR2/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh IDRID_data RETFound_mae 5 RETFound_mae_natureCFP $CKPT_ROOT/IDRID/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh PAPILA RETFound_mae 3 RETFound_mae_natureCFP $CKPT_ROOT/PAPILA/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh Glaucoma_fundus RETFound_mae 3 RETFound_mae_natureCFP $CKPT_ROOT/Glaucoma/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh JSIEC RETFound_mae 39 RETFound_mae_natureCFP $CKPT_ROOT/JSIEC/checkpoint-best.pth --eval
sbatch finetune_retfound_benchmark.sh Retina RETFound_mae 4 RETFound_mae_natureCFP $CKPT_ROOT/Retina/checkpoint-best.pth --eval
#sbatch finetune_retfound_benchmark.sh OCTID RETFound_mae 5 RETFound_mae_natureOCT $CKPT_ROOT/OCTID/checkpoint-best.pth --eval