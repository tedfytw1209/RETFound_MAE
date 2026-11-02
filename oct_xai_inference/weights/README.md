# Weights

The trained weights are **not in git** (they were fit on IRB patient data). A model is one folder here:

```text
weights/<name>/
├── model.pth          {"model": state_dict}, loadable with torch.load(weights_only=True)
└── model_card.json    architecture, class names, input size, default XAI settings
```

`python predict.py --list_models` lists the folders that have a `model_card.json`.

## Adding a trained model

The training scripts save `checkpoint-best.pth`, which also stores the training `args` (model type, input
size, number of classes, and for SMP the fusion settings). The export tool turns that into a model folder,
checks that the weights fit the rebuilt network, and drops the extra training state:

```bash
# RETFound (RETFound_mae, ViT-L/16, 224 px)
python tools/export_checkpoint.py --name retfound_DME --task DME_binary --class_names Control,DME \
    --src /orange/ruogu.fang/tienyuchang/RETfound_results/<run>/checkpoint-best.pth

# SMP-fuse (Unet + resnet50, weighted_sum fusion, 512 px)
python tools/export_checkpoint.py --name smp_fuse_DME --task DME_binary --class_names Control,DME \
    --src /orange/ruogu.fang/tienyuchang/RETfound_results/<run>-fuse-.../checkpoint-best.pth
```

Checkpoints are read with `weights_only=True`, which accepts tensors and the stored `args`. If a checkpoint
needs more than that, the tool stops and asks for `--trust_checkpoint`; use it only for files you trust,
because it unpickles arbitrary objects. `--class_names` is in label order (index 0 first); the training
checkpoint does not store it.
Use `--notes` for cohort / seed / split, and `--xai_method` to change the default CAM method.

You can also skip the export and point `--model` at the raw `checkpoint-best.pth`
(`python predict.py --model .../checkpoint-best.pth --class_names Control,DME ...`).

## model_card.json

Written by the export tool; edit by hand only for checkpoints without stored args (`--card`).
The `//` comments below are explanatory; the real file is plain JSON.

```json
{
  "name": "smp_fuse_DME",
  "arch": "smp",                       // "retfound" or "smp"
  "task": "DME_binary",
  "num_classes": 2,
  "class_names": ["Control", "DME"],   // label order
  "input_size": 512,                   // 224 for RETFound, 512 for SMP
  "weights": "model.pth",
  "smp": {                             // only what differs from the SMP-fuse defaults below is needed
    "seg_arch": "Unet", "encoder": "resnet50", "seg_classes": 9, "seg_activation": "softmax",
    "mode": "fuse", "fuse_mode": "weighted_sum", "learnable_alpha": true, "alpha": 0.5,
    "alpha_type": "scalar", "fusion_dim": 9, "align": "pre", "size_match": "decoder_to_encoder",
    "use_mask": true, "mask_softmax": false, "ignore_background": false,
    "enc_idx": -2, "dec_idx": -1, "smp_classifier": "conv"
  },
  "xai": { "method": "hirescam", "target_module": "head", "select_index": -1 }
}
```

The SMP defaults are the UF SMP configuration of `multiseed_run_all.sh`; the values of a given run are
read from its checkpoint, so a model trained with another gate (`alpha_type`: scalar, channel, spatial,
se, attn) or mode (`enc`, `dec`, `fuse`) needs no manual card. Models trained with `--add_mask` take the
layer-surface masks as an extra input and are not supported here.
