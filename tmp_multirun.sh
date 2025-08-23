

#sbatch infernce_retfound_UFirb2024v4_dualvit.sh AMD_all_split output_dir/AMD_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/AMD_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth

sbatch infernce_retfound_UFirb2024v4_dualvit.sh Cataract_all_split output_dir/Cataract_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/Cataract_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth

sbatch infernce_retfound_UFirb2024v4_dualvit.sh DR_all_split output_dir/DR_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/DR_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth 5e-4 6

sbatch infernce_retfound_UFirb2024v4_dualvit.sh Glaucoma_all_split output_dir/Glaucoma_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/Glaucoma_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth 5e-4 6

sbatch infernce_retfound_UFirb2024v4_dualvit.sh DR_binary_all_split output_dir/DR_binary_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/DR_binary_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth

sbatch infernce_retfound_UFirb2024v4_dualvit.sh Glaucoma_binary_all_split output_dir/Glaucoma_binary_all_split-IRB2024_v4-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth output_dir/Glaucoma_binary_all_split-IRB2024_v4-all-RETFound_mae_natureCFP-CFP-bs16ep50lr5e-4optadamw-roc_auceval--/checkpoint-best.pth