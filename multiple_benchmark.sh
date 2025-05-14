#!/bin/bash

sbatch finetune_retfound_benchmark.sh APTOS2019 retfound 5 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/APTOS2019 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh MESSIDOR2 retfound 5 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/MESSIDOR2 RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh IDRID retfound 5 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/IDRiD_data RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh PAPILA retfound 3 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/PAPILA RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh Glaucoma retfound 3 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/Glaucoma_fundus RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh JSIEC retfound 39 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/JSIEC RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh Retina retfound 4 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/Retina RETFound_mae_natureCFP
sbatch finetune_retfound_benchmark.sh OCTID retfound 5 /orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/OCTID RETFound_mae_natureCFP