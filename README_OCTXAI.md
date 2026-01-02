## OCT XAI - Install, Model Training
### 🔧Install environment

## Training: RETFound Enviroment Install

1. Create environment with conda:

```
conda create -n retfound python=3.11.0 -y
conda activate retfound_new
```

2. Install dependencies

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
git clone https://github.com/rmaphoh/RETFound_MAE/
cd RETFound_MAE
pip install -r requirements_retfound.txt
```

## XAI: OCT_XAI Install

1. Create environment with conda:

```
conda create -n retfound python=3.9 -y
conda activate octxai
```

2. Install dependencies

```
pip install -r requirements.txt
```


### 🌱Fine-tuning with RETFound weights

To fine tune RETFound on your own data, follow these steps:

1. Get access to the pre-trained models on HuggingFace (register an account and fill in the form) and go to step 2:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<th valign="bottom">Source</th>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureCFP</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureCFP">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">RETFound_mae_natureOCT</td>
<td align="center"><a href="https://huggingface.co/YukunZhou/RETFound_mae_natureOCT">access</a></td>
<td align="center"><a href="https://www.nature.com/articles/s41586-023-06555-x">Nature RETFound paper</a></td>
</tr>
</tbody></table>

2. Organise your public dataset into this directory structure (Public datasets used in this study can be [downloaded here](BENCHMARK.md))

```
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
``` 

3. Start fine-tuning (use OCTDL as example) using script

```
sbatch finetune_retfound_OCTDL.sh DME_all RETFound_mae RETFound_mae_natureOCT 1e-3 2 0.05 roc_auc OCT
```
or
```
sbatch finetune_retfound_OCTDL.sh AMD_all vit-base-patch16-224 google/vit-base-patch16-224-in21k 5e-5 2 0.01 roc_auc OCT
```
or
```
sbatch finetune_retfound_OCTDL_smp.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth 1e-4 2 1e-4 default OCT 0 dec
```

Celldata scripts
```
sbatch finetune_retfound_OCTDL.sh DME_all RETFound_mae RETFound_mae_natureOCT 1e-3 2 0.05 roc_auc OCT
```
SMP:
```
sbatch finetune_retfound_Celldata_smp.sh DME_all SMP /blue/ruogu.fang/tienyuchang/RETFound_MAE/Seg_checkpoints/best_model_multiclass.pth 1e-4 2 1e-4 default OCT 0 dec --add_mask --train_no_aug
```

### OCT XAI Case Study and Evaluation

1. For XAI evaluation

Baseline Model: timm_efficientnet-b4, resnet-50, vit-base-patch16-224, RETFound_mae

```
sbatch finetune_retfound_UFbenchmark_v5_eval.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 2 224 hirescam
```

SMP Model

```
sbatch finetune_retfound_UFbenchmark_v5_eval.sh AMD_all_split RETFound_mae RETFound_mae_natureOCT /orange/ruogu.fang/tienyuchang/RETfound_results/DME_binary_all_split-IRB2024_v5-all-RETFound_mae_natureOCT-OCT-bs16ep50lr5e-4optadamw-defaulteval-trsub0--/checkpoint-best.pth 2 224 hirescam
```

2. For Case Study


