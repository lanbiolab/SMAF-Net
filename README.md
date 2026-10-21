# SMAF-Net

Research code for **SMAF-Net: A Scenario-Balanced Mask-Aware Fusion Network for Robust Glaucoma Diagnosis with Incomplete Multimodal Data**.

SMAF-Net combines fundus images, OCT volumes and tabular features using mask-aware attention, scenario-balanced modality dropout and auxiliary supervision.

## Code availability

This repository provides the SMAF-Net implementation, training pipeline, concatenation baseline, and missing-modality evaluation. Datasets, pretrained SMAF-Net checkpoints, and competing-method implementations are not included.

## Contents

| Location | Contents |
| --- | --- |
| `SMAF-Net/models/model.py` | SMAF-Net architecture and mask-aware fusion |
| `SMAF-Net/trainer.py` | Training, scenario sampling, auxiliary losses and validation threshold selection |
| `SMAF-Net/evaluate.py` | Structured and proportional random missingness |
| `Configs/` | UK Biobank experiment configuration |
| `run.py` | Unified command-line entry point |


## Setup and data

Use Python 3.10 or later. Install a compatible PyTorch/torchvision build for your CPU or CUDA environment, then:

```bash
pip install -r requirements.txt
```

Run commands from the repository root and set your data paths in `Configs/smaf_ukb.yaml`. With `pretrained: true`, the image encoders use ImageNet weights, which may download on the first run. Prepare the following inputs:

- `labels.csv`: `Participant ID` and binary `labels` columns.
- `tabular.csv`: `Participant ID` and feature columns.
- Fundus files: `<id>_21015_<suffix>.png` (JPG/JPEG also supported).
- OCT files: `<id>_21017_<suffix>.zip` containing naturally ordered B-scan images.
- `split.json`: disjoint `train_ids`, `val_ids`, `test_ids` lists matching the participant IDs. Keep the split fixed across methods and seeds.

The modality-mask order is `[fundus, oct, tabular]`.

## Training

```bash
python run.py smaf --config Configs/smaf_ukb.yaml
```

For seeds 4 and 6, repeat the SMAF-Net command with a separate output directory and the same participant split:

```bash
python run.py smaf --config Configs/smaf_ukb.yaml --set seed=4 output_dir=outputs/smaf/ALL/seed_4
python run.py smaf --config Configs/smaf_ukb.yaml --set seed=6 output_dir=outputs/smaf/ALL/seed_6
```

Train the concatenation baseline:

```bash
python run.py concat --config Configs/smaf_ukb.yaml --set fusion_type=concat use_mask_aware_fusion=false use_aux_heads=false modality_dropout_rate=0.0 output_dir=outputs/concat/ALL/seed_0
```

For an imaging-only experiment, set `use_tabular=false` and use a separate output directory.

## Missing-modality evaluation

Evaluators discover `ALL/seed_*/checkpoints/best.ckpt` under `--output-root`.

```bash
python run.py missing --config Configs/smaf_ukb.yaml --output-root outputs/smaf --threshold-source val_scenario
python run.py random-missing --config Configs/smaf_ukb.yaml --output-root outputs/smaf --seeds 0 4 6 --repeats 5 --complete-case-only --threshold-source val_ratio
```

Thresholds are selected on validation data. Random masking restores one modality if all are removed, so nominal 100% masking is the single-modality limit. Use `--save-predictions` to save per-participant predictions locally.

## Experiment settings

- `final_logit_mode: fusion` uses the shared fusion classifier for inference. Auxiliary heads provide training supervision.
- Scenario sampling probabilities are defined in `Configs/smaf_ukb.yaml`. Auxiliary loss weights for fundus, OCT and tabular inputs are 0.15, 0.15 and 0.05.
- Supply a participant-level split JSON and keep it fixed across seeds and methods. Original participant splits are not distributed.
- The included configuration targets UK Biobank. GAMMA and Harvard preprocessing and fold-generation scripts are not included.
