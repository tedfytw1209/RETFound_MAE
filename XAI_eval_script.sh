
## Attention-based XAI
sbatch finetune_retfound_UFbenchmark_v2_eval.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT output_dir/AMD_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 attn
sbatch finetune_retfound_UFbenchmark_v2_eval.sh Cataract_all_split RETFound_mae RETFound_mae_natureOCT output_dir/Cataract_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 attn
sbatch finetune_retfound_UFbenchmark_v2_eval.sh DR_binary_all_split RETFound_mae RETFound_mae_natureOCT output_dir/DR_binary_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 attn
sbatch finetune_retfound_UFbenchmark_v2_eval.sh Glaucoma_binary_all_split RETFound_mae RETFound_mae_natureOCT output_dir/Glaucoma_binary_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 attn

## Grad-CAM-based XAI
sbatch finetune_retfound_UFbenchmark_v2_eval.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT output_dir/AMD_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 gradcam
sbatch finetune_retfound_UFbenchmark_v2_eval.sh Cataract_all_split RETFound_mae RETFound_mae_natureOCT output_dir/Cataract_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 gradcam
sbatch finetune_retfound_UFbenchmark_v2_eval.sh DR_binary_all_split RETFound_mae RETFound_mae_natureOCT output_dir/DR_binary_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 gradcam
sbatch finetune_retfound_UFbenchmark_v2_eval.sh Glaucoma_binary_all_split RETFound_mae RETFound_mae_natureOCT output_dir/Glaucoma_binary_all_split-IRB2024_v2-all-RETFound_mae-OCT-mcceval---testval-/checkpoint-best.pth 2 gradcam