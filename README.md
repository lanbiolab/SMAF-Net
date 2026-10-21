# SMAF-Net

Research code for **SMAF-Net: A Scenario-Balanced Mask-Aware Fusion Network for Robust Glaucoma Diagnosis with Incomplete Multimodal Data**.

SMAF-Net combines fundus images, OCT volumes and tabular features using mask-aware attention, scenario-balanced modality dropout and auxiliary supervision.

## Code availability

This source package includes the core SMAF-Net model, training code, a concatenation baseline, and structured and random missing-modality evaluation. Implementations of competing methods are not included. Data, trained weights, participant identifiers and experiment outputs are not distributed. The paper's numerical results have not been independently reproduced from this package.

## Contents

| Location | Contents |
| --- | --- |
| `SMAF-Net/models/model.py` | Mask-aware fusion and multimodal model (legacy class name `BaselineConcatModel`) |
| `SMAF-Net/trainer.py` | Training, scenario sampling, auxiliary losses and validation threshold selection |
| `SMAF-Net/evaluate.py` | Structured and proportional random missingness |
| `Configs/` | Portable UK Biobank input/configuration templates |
| `run.py` | Unified command-line entry point |


## Setup and data

Use Python 3.10 or later. Install a compatible PyTorch/torchvision build for your CPU or CUDA environment, then:

```bash
pip install -r requirements.txt
```

Run commands from this repository root and edit data paths in `Configs/*.yaml`. ImageNet weights may download on the first run (`pretrained: true`). Prepare these local inputs:

- `labels.csv`: `Participant ID` and binary `labels` columns.
- `tabular.csv`: `Participant ID` and feature columns.
- Fundus files: `<id>_21015_<suffix>.png` (JPG/JPEG also supported).
- OCT files: `<id>_21017_<suffix>.zip` containing naturally ordered B-scan images.
- `split.json`: disjoint `train_ids`, `val_ids`, `test_ids` lists matching the participant IDs. Keep the split fixed across methods and seeds.

The modality-mask order is `[fundus, oct, tabular]`. Data and participant identifiers are excluded from version control.

## Training

```bash
python run.py smaf --config Configs/smaf_ukb.yaml
```

For seeds 4 and 6, repeat the SMAF-Net command with a separate output directory and the same participant split:

```bash
python run.py smaf --config Configs/smaf_ukb.yaml --set seed=4 output_dir=outputs/smaf/ALL/seed_4
python run.py smaf --config Configs/smaf_ukb.yaml --set seed=6 output_dir=outputs/smaf/ALL/seed_6
```

Train a concatenation baseline with explicit component switches:

```bash
python run.py concat --config Configs/smaf_ukb.yaml --set fusion_type=concat use_mask_aware_fusion=false use_aux_heads=false modality_dropout_rate=0.0 output_dir=outputs/concat/ALL/seed_0
```

This command defines the baseline settings; it is not a claim that this exact command reproduces a particular reported table row. For an imaging-only comparison, also set `use_tabular=false` and use a separate output directory.

## Missing-modality evaluation

Evaluators discover `ALL/seed_*/checkpoints/best.ckpt` under `--output-root`.

```bash
python run.py missing --config Configs/smaf_ukb.yaml --output-root outputs/smaf --threshold-source val_scenario
python run.py random-missing --config Configs/smaf_ukb.yaml --output-root outputs/smaf --seeds 0 4 6 --repeats 5 --complete-case-only --threshold-source val_ratio
```

Thresholds are selected on validation data. Random masking restores one modality if all are removed, so nominal 100% masking is the single-modality limit. `--save-predictions` writes participant-level predictions locally; do not publish these with the repository.

## Reproducibility notes

- The supplied SMAF-Net configuration explicitly sets `final_logit_mode: fusion`: auxiliary heads supervise training; inference uses the shared fusion classifier.
- Scenario probabilities (0.20/0.10/0.10/0.20/0.15/0.15/0.10) and auxiliary weights (0.15/0.15/0.05) follow the archived fusion-output experiment summary. Scenario-balanced does not mean uniform sampling.
- Original participant splits are excluded. Supply your authorized split JSON and preserve it across seeds and methods. Exact results depend on the original cohort, preprocessing, split and environment.
- Configurations are UK Biobank templates. GAMMA/Harvard preprocessing, folds and exact table reproduction are not packaged here.
- Legacy code contains optional inference modes and experiments beyond the manuscript. Use explicit configuration flags rather than constructor defaults.
- Source is curated from the authors' SMAF-Net experiment project. Syntax and configuration checks have been performed; full training has not been revalidated in this package.
