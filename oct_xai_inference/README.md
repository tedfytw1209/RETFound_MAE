# OCT inference: RETFound and SMP-fuse (probabilities + saliency maps)

Standalone inference for the two trained OCT classifiers of this repository:

- **RETFound** (`RETFound_mae`, ViT-L/16, 224 px input)
- **SMP-fuse** (`SMP`: Unet + resnet50 segmentation network whose encoder and decoder features are fused by
  a weighted-sum gate and a 1x1 conv classifier, 512 px input)

For every input image it writes the class probabilities and an XAI saliency map (Grad-CAM family).

This folder imports nothing from the rest of the repository; the model code is vendored (see
[Independence](#independence)). Copy it anywhere. The layout follows `ml-classification/oct_ad_inference`.

> **Research use only.** Not a diagnostic device. Outputs are model scores, and a saliency map shows where
> the model looked, not a verified lesion location.

---

## Quick start

```bash
cd oct_xai_inference
pip install -r requirements.txt          # on HiPerGator: module load conda && conda activate octxai

# 1. put a trained model in weights/ (see weights/README.md)
python tools/export_checkpoint.py --name smp_fuse_DME --task DME_binary --class_names Control,DME \
    --src /orange/ruogu.fang/tienyuchang/RETfound_results/<run>/checkpoint-best.pth

# 2. predict: an image, a directory of images, or a CSV of image paths (any mix)
python predict.py --model smp_fuse_DME --input scan.jpg --output preds.csv
python predict.py --model retfound_DME --input bscans/ other.csv --output out/preds.csv

# probabilities only (no saliency)
python predict.py --model smp_fuse_DME --input bscans/ --no_saliency

# list models in weights/
python predict.py --list_models
```

Or use the wrapper, which also works with `sbatch` on HiPerGator:

```bash
MODEL=smp_fuse_DME bash run_inference.sh scan.jpg
MODEL=retfound_DME OUTPUT=out/retfound.csv sbatch run_inference.sh bscans/
```

To score the same images with both models, run the command once per model with different `--output` paths.

---

## Directory layout

```
oct_xai_inference/
├── predict.py                 # CLI entry point
├── run_inference.sh           # bash/sbatch wrapper (configured by env vars)
├── requirements.txt
├── octxai/                    # library (importable)
│   ├── model.py               # model cards, building the network, loading weights
│   ├── vit.py                 # RETFound ViT-L           (vendored from models_vit.py)
│   ├── smp_classifier.py      # SMP classifier           (vendored from SMP/smp_classifier.py)
│   ├── fusion_head.py         # fusion head + gates      (vendored from SMP/special_header.py)
│   ├── preprocess.py          # image loading + the evaluation transform
│   ├── xai.py                 # HiResCAM / GradCAM / GradCAM++ (as in baselines/GradCAM_v2.py)
│   ├── viz.py                 # overlay PNGs
│   └── pipeline.py            # inputs -> probabilities -> saliency -> output table
├── weights/                   # trained models, NOT in git (see weights/README.md)
├── tools/export_checkpoint.py # package a training checkpoint as weights/<name>/
└── tests/                     # pytest: behaviour with throwaway models + parity with the training code
```

---

## Input format

Any mix of:

| Input | Notes |
|---|---|
| image file | `.jpg .jpeg .png .bmp .tif .tiff`. Grayscale B-scans are converted to RGB, as in training. |
| directory | all images directly inside it; `--recursive` to descend. `image_name` is the path relative to the directory. |
| `.csv` | one row per image. Path column: `--path_col`, else the first of `image_path, path, filepath, image, OCT`. Relative paths resolve against `--image_root`. Names without extension get `.jpg`, as in the training CSVs (`OCT` column + `--img_dir`). An `image_name` column, if present, becomes the output `image_name`. |

Each image is one OCT B-scan (the models were trained on single B-scans of the macula). Preprocessing is
the training-time evaluation transform: bicubic resize and centre crop to the model's input size
(224 for RETFound with the usual 256 -> 224 crop, 512 for SMP), then ImageNet normalisation. Because the
resize scales the shorter side, a wide B-scan is cropped along its width, and the saliency map and
overlay cover that cropped region.

## Output

`--output preds.csv` has one row per input, including images that could not be read:

| Column | Meaning |
|---|---|
| `image_name` | image identifier (see above) |
| `label_0_prob`, `label_1_prob`, ... | softmax probability per class (index = training label order) |
| `pred_label` | binary: 1 if `label_1_prob >= --threshold` (default 0.5); otherwise argmax |
| `pred_class` | class name from the model card |
| `xai_class` | the class the saliency map explains |
| `status` | `ok`; `ok; saliency failed (...)`; or `skipped: cannot read image (...)` (probabilities left empty) |
| `source_file` | which input file / directory the image came from |
| `saliency_npy`, `saliency_png` | paths, relative to the output CSV's directory |

Saliency files go to `<output without .csv>_saliency/` (`--saliency_dir` to change):

- `npy/<image>.npy`: float32 map in [0, 1], shape `(input_size, input_size)`, aligned with the network
  input (the cropped, resized image, not the original file).
- `png/<image>.png`: the network input on the left, the same image blended with the jet-coloured map on
  the right. `--no_overlay` skips these.

---

## Saliency (XAI)

The methods and layers are those of the XAI evaluation (`main_XAI_evaluation.py`, `baselines/GradCAM_v2.py`),
and the tests check that they give the same maps as that code.

| Option | Default | Notes |
|---|---|---|
| `--xai` | `hirescam` | `hirescam`, `gradcamv2` (Grad-CAM), `gradcam++`. The three methods of the multi-seed evaluation. |
| `--xai_target` | `pred` | the class to explain: the predicted class, or an index (`0`, `1`) to explain a given class. The evaluation explains the ground-truth class, which is unknown at inference. |
| `--xai_module`, `--xai_index` | from the card | RETFound: transformer block `blocks[-1]`, its tokens reshaped to a 14 x 14 grid. SMP: the last `Conv2d` of the `head`, i.e. the 1x1 classifier on the fused features (as in `--target_module head --select_index -1`). `encoder` / `decoder` hook those parts of the segmentation network instead. |

Maps are min-max normalised per image and bilinearly resized to the input resolution.

---

## CLI reference

```
python predict.py -h
```

| Arg | Default | Description |
|---|---|---|
| `--input, -i` | *required* | Images, directories and/or CSVs. |
| `--output, -o` | `predictions.csv` | Output CSV (parent dirs created). |
| `--model, -m` | *required* | Name under `weights/`, a model directory, or a training `checkpoint-best.pth`. |
| `--list_models` | | List `weights/` and exit. |
| `--class_names` | from card | `Control,DME`; needed for raw training checkpoints, which do not store them. |
| `--threshold` | `0.5` | Binary decision threshold on `label_1_prob`. |
| `--device` | `auto` | `auto` (GPU if available), `cpu`, `cuda:1`, ... |
| `--batch_size` | `8` | Images per forward pass. CAM methods backpropagate, so lower it if the GPU runs out of memory. |
| `--n_workers` | `4` | Threads that read images. |
| `--no_saliency` | off | Probabilities only (no `grad-cam` needed). |
| `--xai`, `--xai_target`, `--xai_module`, `--xai_index` | | See above. |
| `--no_overlay` | off | Save `.npy` maps only. |
| `--saliency_dir` | `<output>_saliency/` | Where saliency files go. |
| `--recursive`, `--path_col`, `--image_root` | | Input handling, see above. |
| `--trust_checkpoint` | off | Allow checkpoints that need full pickle (trusted files only). |
| `--verbose, -v` | | Progress. |

---

## Independence

The code that defines the networks was copied, not imported, so this folder runs without the training
repository: `octxai/vit.py` <- `models_vit.py`, `octxai/smp_classifier.py` <- `SMP/smp_classifier.py`,
`octxai/fusion_head.py` <- `SMP/special_header.py`. Module and parameter names are unchanged, so the
fine-tuned checkpoints load by key, and loading is strict: a card that does not match the weights is an
error, not a silent partial load.

Two deliberate differences from the training code:

- **timm.** The RETFound checkpoints were trained with timm 0.9.x, whose `forward_head` takes token 0 and
  applies `fc_norm` a second time. timm >= 1.0 pools differently (and the original `models_vit.py` fails to
  run on it), so `octxai/vit.py` pins the 0.9.x behaviour and gives the same outputs on either version.
- **Attention path.** `TIMM_FUSED_ATTN=0` is set by default, as in the XAI-evaluation jobs.

If you change the vendored model code in the repository, copy the change here and run the parity tests.

## Tests

```bash
pip install -r requirements.txt
pytest tests -q
```

`tests/test_inference.py` runs the CLI on throwaway random-weight models (tiny architectures, synthetic
images; no patient data). `tests/test_parity.py` compares the vendored networks and CAM maps with the
originals in the surrounding repository; it is skipped when this folder is used on its own, and the ViT
comparison needs `timm==0.9.12` because the original class does not run on newer timm.

Checked on CPU with torch 2.13: all tests pass under timm 0.9.12 (including parity), and with timm 1.0.28
all but the ViT parity comparison, which skips. The full-size models (ViT-L 224 px, resnet50 SMP-fuse 512 px,
random weights) also run through `export_checkpoint.py` and `predict.py` end to end. No trained checkpoint
was available when this was written, so the match to real results rests on the parity tests.
