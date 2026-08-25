# ViT Compression for Agricultural Disease Detection — Reproduction

This repository contains the code and results for:

> An Empirical Study of Pruning and Quantization for Vision Transformer Compression in Agricultural Disease Detection

## Environment

- Python 3.10+, PyTorch >= 2.3, timm, torchvision, safetensors
- GPU with >= 8 GB VRAM recommended (RTX 5060 used for agricultural experiments)

```bash
pip install -r requirements.txt
```

## Paths (placeholders)

Set the following constants in the scripts before running:

| Placeholder | Meaning |
|---|---|
| `REPO_ROOT` | Absolute path to this repository root |
| `DATASETS/plantvillage` | PlantVillage augmented Kaggle release (`train`/`valid` subfolders) |
| `DATASETS/plantdoc` | PlantDoc dataset (`train` subfolder) |
| `PRETRAINED_VIT` | Path to the timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` safetensors weights |

## Data

| Dataset | Source | Usage |
|---|---|---|
| PlantVillage (augmented Kaggle release) | [Kaggle: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) | 70,295 train / 17,572 valid, 38 classes |
| PlantDoc | [GitHub: PlantDoc](https://github.com/pratikrm/PlantDoc) | 2,336 images, 28 classes; 80/20 split with seed 42 |
| ImageNet-1K | Standard | 1.28M train / 50K valid, 1,000 classes |

Pre-trained ViT-B/16 weights: `timm vit_base_patch16_224.augreg2_in21k_ft_in1k`.

## Experiments

| Script | Experiment |
|---|---|
| `run_m4_seeds.py` | PlantVillage main protocol: 3 seeds x (baseline FT, prune 25/50/75% +/- FT) |
| `run_m4_plantdoc.py` | PlantDoc main protocol, 3 seeds |
| `run_m4_headkeep.py` / `run_m4_headkeep_pd.py` | Trained-head no-FT control (isolates pruning damage from untrained-head artifact) |
| `run_m5_int8.py` | INT8 dynamic quantization evaluation (CPU) |
| `run_pd_combo.py` | Combined prune + FT + INT8 on PlantDoc (supplementary experiment) |
| `run_cnn_baseline.py` | Lightweight CNN baselines (MobileNetV3-Large/Small) |
| `build_results_master.py` | Aggregates all JSON outputs into `results_master.csv` |

Run order: main protocols -> quantization/combo -> `build_results_master.py`.

## Protocol (unified for all agricultural experiments)

- Optimizer: AdamW, lr 5e-5 (ViT) / 1e-3 (CNN), weight decay 1e-2 / 1e-4
- 5 epochs, cosine LR schedule, AMP (fp16), batch 32 (ViT) / 64 (CNN)
- Input: resize 256x256 -> center-crop 224x224; train adds random horizontal flip
- Seeds: 42, 123, 2026; PlantDoc split fixed with seed 42 (80/20)
- Pruning: sensitivity-guided zeroing of attention-head weights (qkv/proj), 25/50/75%
- INT8: `torch.ao.quantization.quantize_dynamic` on linear layers, evaluated on CPU

## Results

`results_master.csv` is the single source of truth for all reported numbers (accuracy, latency, peak memory). See the manuscript Table 3 for the aggregated view.
