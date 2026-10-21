"""Structured and proportional random missing-modality evaluation."""
from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from trainer import _build_loaders, load_model_from_checkpoint
from utils import calculate_metrics, find_best_threshold


structured_DEFAULT_SCENARIOS = [
    "natural",
    "full_complete",
    "no_fundus",
    "no_oct",
    "no_tabular",
    "fundus_only",
    "oct_only",
    "tabular_only",
]


def structured_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate missing-modality robustness across multiple seed checkpoints. "
            "Supports scenario-specific validation thresholds."
        )
    )
    parser.add_argument("--config", type=str, default="config_baseline_concat.yaml")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory that contains ALL/seed_*/checkpoints/best.ckpt",
    )
    parser.add_argument("--experiment", type=str, default="ALL")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for eval batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Optional override for eval num_workers",
    )
    parser.add_argument(
        "--use-natural-missing",
        action="store_true",
        help="Force allow_missing_modalities=True when building eval loaders",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=structured_DEFAULT_SCENARIOS,
        choices=structured_DEFAULT_SCENARIOS,
        help="Scenarios to evaluate",
    )
    parser.add_argument(
        "--threshold-source",
        type=str,
        default="val_scenario",
        choices=["val_scenario", "summary", "fixed"],
        help=(
            "val_scenario: for each seed/scenario, tune threshold on val split under the same scenario, "
            "then evaluate the requested split. "
            "summary: use train_summary best_threshold_from_val. "
            "fixed: use --fixed-threshold."
        ),
    )
    parser.add_argument(
        "--threshold-metric",
        type=str,
        default="F1",
        help="Metric used by find_best_threshold when --threshold-source val_scenario.",
    )
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions for each scenario/seed",
    )
    return parser.structured_parse_args()


def structured_discover_checkpoints(output_root: Path, experiment: str) -> List[Path]:
    ckpts = sorted(
        (output_root / experiment).glob("seed_*/checkpoints/best.ckpt"),
        key=lambda p: p.as_posix(),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {output_root / experiment}")
    return ckpts


def structured_seed_from_checkpoint(path: Path) -> int:
    try:
        return int(path.parent.parent.name.split("_")[-1])
    except Exception:
        return -1


def structured_load_summary_threshold(checkpoint_path: Path, fixed_threshold: float) -> float:
    summary_path = checkpoint_path.parent.parent / "train_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        thr = summary.get("best_threshold_from_val")
        if thr is not None:
            return float(thr)

    return float(fixed_threshold)


def structured_pick_loader(split: str, loaders: Tuple):
    train_loader, val_loader, test_loader = loaders[:3]
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def structured_apply_scenario(orig_mask: torch.Tensor, scenario: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        keep_rows: [B] bool, rows retained for evaluation.
        final_mask: [B, 3] bool, mask passed to model.

    Note:
        natural:
            use the original modality availability.

        forced-missing scenarios:
            evaluate only on complete-case samples.
            This ensures all forced-missing scenarios are compared on the same cohort.
    """
    if orig_mask.ndim != 2 or orig_mask.size(1) != 3:
        raise ValueError(f"Expected modality mask shape [B,3], got {tuple(orig_mask.shape)}")

    orig_mask = orig_mask.bool()

    if scenario == "natural":
        keep_rows = orig_mask.any(dim=1)
        final_mask = orig_mask.clone()
        return keep_rows, final_mask

    keep_rows = orig_mask.all(dim=1)
    base = torch.ones_like(orig_mask, dtype=torch.bool)

    if scenario == "full_complete":
        final_mask = base
    elif scenario == "no_fundus":
        final_mask = base
        final_mask[:, 0] = False
    elif scenario == "no_oct":
        final_mask = base
        final_mask[:, 1] = False
    elif scenario == "no_tabular":
        final_mask = base
        final_mask[:, 2] = False
    elif scenario == "fundus_only":
        final_mask = torch.zeros_like(orig_mask, dtype=torch.bool)
        final_mask[:, 0] = True
    elif scenario == "oct_only":
        final_mask = torch.zeros_like(orig_mask, dtype=torch.bool)
        final_mask[:, 1] = True
    elif scenario == "tabular_only":
        final_mask = torch.zeros_like(orig_mask, dtype=torch.bool)
        final_mask[:, 2] = True
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    keep_rows = keep_rows & final_mask.any(dim=1)
    return keep_rows, final_mask


@torch.no_grad()
def structured_collect_scenario_predictions(
    model,
    loader,
    device: torch.device,
    scenario: str,
    desc: str,
) -> Dict:
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_pids: List[str] = []
    n_seen = 0
    n_kept = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        if len(batch) != 6:
            raise ValueError(
                "Expected batch length 6. "
                "Make sure dataset.py and trainer.py return modality_mask."
            )

        fundus, oct_volume, tab, label, pid, modality_mask = batch
        modality_mask = modality_mask.bool()

        keep_rows, forced_mask = structured_apply_scenario(modality_mask, scenario)
        n_seen += int(modality_mask.size(0))
        n_kept += int(keep_rows.sum().item())

        if keep_rows.sum().item() == 0:
            continue

        fundus = fundus[keep_rows].to(device, non_blocking=True)
        oct_volume = oct_volume[keep_rows].to(device, non_blocking=True)
        tab = tab[keep_rows].to(device, non_blocking=True)
        label = label[keep_rows].to(device, non_blocking=True)
        forced_mask = forced_mask[keep_rows].to(device, non_blocking=True)
        pid = [pid[i] for i, keep in enumerate(keep_rows.tolist()) if keep]

        outputs = model(fundus, oct_volume, tab, forced_mask)
        probs = outputs["prob"].detach().cpu().numpy().astype(np.float64)
        labels = label.detach().cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_pids.extend(pid)

    return {
        "probs": np.asarray(all_probs, dtype=np.float64),
        "labels": np.asarray(all_labels, dtype=int),
        "pids": all_pids,
        "num_seen": int(n_seen),
        "num_eval": int(n_kept),
    }


def structured_determine_threshold(
    threshold_source: str,
    fixed_threshold: float,
    checkpoint_path: Path,
    model,
    val_loader,
    device: torch.device,
    scenario: str,
    threshold_metric: str,
    seed: int,
) -> Tuple[float, Dict]:
    """
    Returns:
        threshold
        threshold_info dict with val-threshold diagnostics
    """
    if threshold_source == "fixed":
        return float(fixed_threshold), {
            "threshold_source": "fixed",
            "val_threshold_auc": np.nan,
            "val_threshold_acc": np.nan,
            "val_threshold_f1": np.nan,
            "val_num_eval_for_threshold": np.nan,
        }

    if threshold_source == "summary":
        return structured_load_summary_threshold(checkpoint_path, fixed_threshold), {
            "threshold_source": "summary",
            "val_threshold_auc": np.nan,
            "val_threshold_acc": np.nan,
            "val_threshold_f1": np.nan,
            "val_num_eval_for_threshold": np.nan,
        }

    # threshold_source == "val_scenario"
    val_pred = structured_collect_scenario_predictions(
        model=model,
        loader=val_loader,
        device=device,
        scenario=scenario,
        desc=f"seed={seed} {scenario} tune-val",
    )

    if len(val_pred["probs"]) == 0:
        return float(fixed_threshold), {
            "threshold_source": "val_scenario_empty_fallback_fixed",
            "val_threshold_auc": np.nan,
            "val_threshold_acc": np.nan,
            "val_threshold_f1": np.nan,
            "val_num_eval_for_threshold": int(val_pred["num_eval"]),
        }

    threshold = find_best_threshold(
        val_pred["labels"],
        val_pred["probs"],
        metric_name=threshold_metric,
    )

    val_metrics = calculate_metrics(
        val_pred["labels"],
        val_pred["probs"],
        threshold=float(threshold),
    )

    return float(threshold), {
        "threshold_source": "val_scenario",
        "val_threshold_auc": float(val_metrics.get("AUROC", np.nan)),
        "val_threshold_acc": float(val_metrics.get("Acc", np.nan)),
        "val_threshold_f1": float(val_metrics.get("F1", np.nan)),
        "val_num_eval_for_threshold": int(val_pred["num_eval"]),
    }


@torch.no_grad()
def structured_evaluate_checkpoint_scenarios(
    checkpoint_path: Path,
    config_path: Path,
    split: str,
    scenarios: Iterable[str],
    threshold_source: str,
    threshold_metric: str,
    fixed_threshold: float,
    device_override: str | None,
    batch_size_override: int | None,
    num_workers_override: int | None,
    use_natural_missing: bool,
    save_predictions: bool,
    pred_out_dir: Path,
) -> List[Dict]:
    model, cfg, device, _ = load_model_from_checkpoint(
        str(config_path),
        str(checkpoint_path),
        device=device_override,
    )

    if batch_size_override is not None:
        cfg["batch_size"] = int(batch_size_override)
    if num_workers_override is not None:
        cfg["num_workers"] = int(num_workers_override)
    if use_natural_missing:
        cfg["allow_missing_modalities"] = True

    loaders = _build_loaders(cfg)
    train_loader, val_loader, test_loader = loaders[:3]
    eval_loader = structured_pick_loader(split, loaders)

    seed = structured_seed_from_checkpoint(checkpoint_path)
    rows: List[Dict] = []

    for scenario in scenarios:
        threshold, threshold_info = structured_determine_threshold(
            threshold_source=threshold_source,
            fixed_threshold=fixed_threshold,
            checkpoint_path=checkpoint_path,
            model=model,
            val_loader=val_loader,
            device=device,
            scenario=scenario,
            threshold_metric=threshold_metric,
            seed=seed,
        )

        pred = structured_collect_scenario_predictions(
            model=model,
            loader=eval_loader,
            device=device,
            scenario=scenario,
            desc=f"seed={seed} {scenario} eval-{split}",
        )

        if len(pred["probs"]) == 0:
            rows.append({
                "seed": seed,
                "scenario": scenario,
                "split": split,
                "num_seen": int(pred["num_seen"]),
                "num_eval": int(pred["num_eval"]),
                "threshold": float(threshold),
                **threshold_info,
                "AUC": np.nan,
                "ACC": np.nan,
                "F1": np.nan,
                "checkpoint": checkpoint_path.as_posix(),
            })
            continue

        metrics = calculate_metrics(
            pred["labels"],
            pred["probs"],
            threshold=float(threshold),
        )

        rows.append({
            "seed": seed,
            "scenario": scenario,
            "split": split,
            "num_seen": int(pred["num_seen"]),
            "num_eval": int(pred["num_eval"]),
            "threshold": float(threshold),
            **threshold_info,
            "AUC": float(metrics.get("AUROC", np.nan)),
            "ACC": float(metrics.get("Acc", np.nan)),
            "F1": float(metrics.get("F1", np.nan)),
            "checkpoint": checkpoint_path.as_posix(),
        })

        if save_predictions:
            pred_df = pd.DataFrame({
                "Participant ID": pred["pids"],
                "label": pred["labels"].astype(int),
                "prob": pred["probs"].astype(np.float64),
                "pred": (pred["probs"] >= float(threshold)).astype(int),
                "threshold": float(threshold),
                "seed": seed,
                "scenario": scenario,
                "split": split,
            })
            pred_path = pred_out_dir / f"pred_{split}_{scenario}_seed_{seed}.csv"
            pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return rows


def structured_summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["AUC", "ACC", "F1", "threshold"]

    for c in metric_cols + ["num_seen", "num_eval", "val_num_eval_for_threshold"]:
        if c in raw_df.columns:
            raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

    grouped = raw_df.groupby("scenario", as_index=False).agg(
        seeds=("seed", "count"),
        num_eval_mean=("num_eval", "mean"),
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        ACC_mean=("ACC", "mean"),
        ACC_std=("ACC", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
    )

    scenario_order = {name: i for i, name in enumerate(structured_DEFAULT_SCENARIOS)}
    grouped["_order"] = grouped["scenario"].map(scenario_order).fillna(999)
    grouped = grouped.sort_values(["_order", "scenario"]).drop(columns=["_order"])
    return grouped


def structured_format_mean_std(mean_val, std_val) -> str:
    if pd.isna(mean_val):
        return ""
    if pd.isna(std_val):
        return f"{mean_val:.4f}"
    return f"{mean_val:.4f} ± {std_val:.4f}"


def structured_write_summary_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Missing Modality Evaluation Summary")
    lines.append("")
    lines.append("| Scenario | Threshold | Test AUC | Test ACC | Test F1 |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, row in summary_df.iterrows():
        lines.append(
            "| {scenario} | {thr} | {auc} | {acc} | {f1} |".format(
                scenario=row["scenario"],
                thr=structured_format_mean_std(row["threshold_mean"], row["threshold_std"]),
                auc=structured_format_mean_std(row["AUC_mean"], row["AUC_std"]),
                acc=structured_format_mean_std(row["ACC_mean"], row["ACC_std"]),
                f1=structured_format_mean_std(row["F1_mean"], row["F1_std"]),
            )
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def structured_main() -> None:
    args = structured_parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    ckpts = structured_discover_checkpoints(output_root, args.experiment)

    eval_root = output_root / f"{args.experiment}_missing_eval_{args.split}_{args.threshold_source}"
    eval_root.mkdir(parents=True, exist_ok=True)

    pred_out_dir = eval_root / "predictions"
    if args.save_predictions:
        pred_out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for ckpt_path in ckpts:
        rows = structured_evaluate_checkpoint_scenarios(
            checkpoint_path=ckpt_path,
            config_path=config_path,
            split=args.split,
            scenarios=args.scenarios,
            threshold_source=args.threshold_source,
            threshold_metric=args.threshold_metric,
            fixed_threshold=args.fixed_threshold,
            device_override=args.device,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            use_natural_missing=args.use_natural_missing,
            save_predictions=args.save_predictions,
            pred_out_dir=pred_out_dir,
        )
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    summary_df = structured_summarize(raw_df.copy())

    raw_csv = eval_root / "missing_modality_raw.csv"
    summary_csv = eval_root / "missing_modality_summary.csv"
    summary_md = eval_root / "missing_modality_summary.md"

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    structured_write_summary_markdown(summary_df, summary_md)

    meta = {
        "config": str(config_path),
        "output_root": str(output_root),
        "experiment": args.experiment,
        "split": args.split,
        "num_checkpoints": len(ckpts),
        "scenarios": list(args.scenarios),
        "threshold_source": args.threshold_source,
        "threshold_metric": args.threshold_metric,
        "fixed_threshold": float(args.fixed_threshold),
        "use_natural_missing": bool(args.use_natural_missing),
        "raw_csv": raw_csv.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "summary_md": summary_md.as_posix(),
    }

    (eval_root / "missing_modality_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("========== Done ==========")
    print(f"raw_csv: {raw_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_md: {summary_md}")
    print(f"threshold_source: {args.threshold_source}")
    print(f"threshold_metric: {args.threshold_metric}")




##之前用的是5-25的旧eval_proportional_random_missing

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from trainer import _build_loaders, load_model_from_checkpoint
from utils import calculate_metrics, find_best_threshold


def random_parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate proportional random missing modality robustness curves. "
            "At each missing ratio p, each available modality is randomly masked with probability p, "
            "while ensuring at least one modality remains available. "
            "This version supports seed filtering and physical zeroing of masked inputs, "
            "which is important for Concat baselines that may not use modality_mask internally."
        )
    )
    parser.add_argument("--config", type=str, default="config_baseline_concat.yaml")
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory that contains ALL/seed_*/checkpoints/best.ckpt",
    )
    parser.add_argument("--experiment", type=str, default="ALL")
    parser.add_argument("--model-name", type=str, default="Ours")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        help="Missing modality ratios.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Random repeats per seed and ratio.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional seed filter. Example: --seeds 0 4 6. "
            "If omitted, all discovered seed_* checkpoints are evaluated."
        ),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold-source",
        type=str,
        default="val_ratio",
        choices=["val_ratio", "summary", "fixed"],
        help=(
            "val_ratio: tune threshold on validation set under the same missing ratio and repeat; "
            "summary: use best_threshold_from_val from train_summary.json; "
            "fixed: use --fixed-threshold."
        ),
    )
    parser.add_argument("--threshold-metric", type=str, default="F1")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument(
        "--complete-case-only",
        action="store_true",
        help="Evaluate only samples whose original modality_mask has all three modalities. Recommended for fair comparison.",
    )
    parser.add_argument(
        "--zero-masked-inputs",
        dest="zero_masked_inputs",
        action="store_true",
        default=True,
        help=(
            "Physically set masked modalities to zero before model forward. "
            "Default: enabled. This is required for fair evaluation of Concat baselines."
        ),
    )
    parser.add_argument(
        "--no-zero-masked-inputs",
        dest="zero_masked_inputs",
        action="store_false",
        help="Disable physical zeroing and only pass the final modality_mask to the model.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: <output-root>/<experiment>_random_missing_curve_<split>_<threshold-source>",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.random_parse_args()


def random_discover_checkpoints(output_root: Path, experiment: str, seeds: List[int] | None = None) -> List[Path]:
    ckpts = sorted(
        (output_root / experiment).glob("seed_*/checkpoints/best.ckpt"),
        key=lambda p: p.as_posix(),
    )
    if seeds is not None:
        seed_set = {int(s) for s in seeds}
        ckpts = [p for p in ckpts if random_seed_from_checkpoint(p) in seed_set]

    if not ckpts:
        if seeds is None:
            raise FileNotFoundError(f"No checkpoints found under: {output_root / experiment}")
        raise FileNotFoundError(
            f"No checkpoints found under {output_root / experiment} for seeds: {seeds}"
        )
    return ckpts


def random_seed_from_checkpoint(path: Path) -> int:
    try:
        return int(path.parent.parent.name.split("_")[-1])
    except Exception:
        return -1


def random_load_summary_threshold(checkpoint_path: Path, fixed_threshold: float) -> float:
    summary_path = checkpoint_path.parent.parent / "train_summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        thr = summary.get("best_threshold_from_val")
        if thr is not None:
            return float(thr)
    return float(fixed_threshold)


def random_pick_loader(split: str, loaders: Tuple):
    train_loader, val_loader, test_loader = loaders[:3]
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def random_apply_random_missing(
    orig_mask: torch.Tensor,
    ratio: float,
    generator: torch.Generator,
    complete_case_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly drop each available modality with probability ratio.

    Args:
        orig_mask: [B, 3], bool. Order must be [fundus, oct, tabular].
        ratio: probability of masking each originally available modality.
        generator: torch random generator for reproducible masking.
        complete_case_only: if True, evaluate only original full-modality samples.

    Returns:
        keep_rows: [B] bool, rows retained for evaluation.
        final_mask: [B, 3] bool, final mask passed to the model.

    Important:
        If all modalities are dropped for a retained sample, randomly keep one originally available modality.
    """
    if orig_mask.ndim != 2 or orig_mask.size(1) != 3:
        raise ValueError(f"Expected modality_mask shape [B,3], got {tuple(orig_mask.shape)}")

    orig_mask = orig_mask.bool()
    device = orig_mask.device
    ratio = float(ratio)

    if complete_case_only:
        keep_rows = orig_mask.all(dim=1)
    else:
        keep_rows = orig_mask.any(dim=1)

    if ratio <= 0:
        final_mask = orig_mask.clone()
        return keep_rows, final_mask

    if ratio >= 1.0:
        final_mask = torch.zeros_like(orig_mask, dtype=torch.bool)
    else:
        rand = torch.rand(orig_mask.shape, device=device, generator=generator)
        keep_random = rand > ratio
        final_mask = orig_mask & keep_random

    empty_rows = (final_mask.sum(dim=1) == 0) & keep_rows
    if empty_rows.any():
        empty_indices = torch.where(empty_rows)[0]
        for idx in empty_indices:
            available = torch.where(orig_mask[idx])[0]
            if available.numel() > 0:
                chosen_pos = torch.randint(
                    low=0,
                    high=available.numel(),
                    size=(1,),
                    device=device,
                    generator=generator,
                )
                chosen_modality = available[chosen_pos]
                final_mask[idx, chosen_modality] = True

    keep_rows = keep_rows & final_mask.any(dim=1)
    return keep_rows, final_mask


def random_zero_masked_modalities(
    fundus: torch.Tensor,
    oct_volume: torch.Tensor,
    tab: torch.Tensor,
    final_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Physically remove masked modalities by setting their inputs to zero.

    This is critical for Concat / early-fusion baselines that may not use modality_mask internally.
    Mask order is fixed as [fundus, oct, tabular].
    """
    final_mask = final_mask.bool()

    fundus_keep = final_mask[:, 0].float().view(-1, 1, 1, 1)
    oct_keep = final_mask[:, 1].float().view(-1, 1, 1, 1, 1)
    tab_keep = final_mask[:, 2].float().view(-1, 1)

    fundus = fundus * fundus_keep
    oct_volume = oct_volume * oct_keep
    tab = tab * tab_keep
    return fundus, oct_volume, tab


@torch.no_grad()
def random_collect_predictions_random_missing(
    model,
    loader,
    device: torch.device,
    ratio: float,
    random_seed: int,
    complete_case_only: bool,
    zero_masked_inputs: bool,
    desc: str,
) -> Dict:
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_pids: List[str] = []
    n_seen = 0
    n_eval = 0

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_seed))

    for batch in tqdm(loader, desc=desc, leave=False):
        if len(batch) != 6:
            raise ValueError("Expected batch length 6. Make sure dataset returns modality_mask.")

        fundus, oct_volume, tab, label, pid, modality_mask = batch
        modality_mask = modality_mask.bool().to(device, non_blocking=True)

        keep_rows, final_mask = random_apply_random_missing(
            orig_mask=modality_mask,
            ratio=ratio,
            generator=generator,
            complete_case_only=complete_case_only,
        )

        n_seen += int(modality_mask.size(0))
        n_eval += int(keep_rows.sum().item())

        if keep_rows.sum().item() == 0:
            continue

        keep_cpu = keep_rows.detach().cpu()

        fundus = fundus[keep_cpu].to(device, non_blocking=True)
        oct_volume = oct_volume[keep_cpu].to(device, non_blocking=True)
        tab = tab[keep_cpu].to(device, non_blocking=True)
        label = label[keep_cpu].to(device, non_blocking=True)
        final_mask = final_mask[keep_rows].to(device, non_blocking=True)
        pid_kept = [pid[i] for i, keep in enumerate(keep_cpu.tolist()) if keep]

        if zero_masked_inputs:
            fundus, oct_volume, tab = random_zero_masked_modalities(
                fundus=fundus,
                oct_volume=oct_volume,
                tab=tab,
                final_mask=final_mask,
            )

        outputs = model(fundus, oct_volume, tab, final_mask)
        probs = outputs["prob"].detach().cpu().numpy().astype(np.float64)
        labels = label.detach().cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())
        all_pids.extend(pid_kept)

    return {
        "probs": np.asarray(all_probs, dtype=np.float64),
        "labels": np.asarray(all_labels, dtype=int),
        "pids": all_pids,
        "num_seen": int(n_seen),
        "num_eval": int(n_eval),
    }


def random_determine_threshold(
    threshold_source: str,
    fixed_threshold: float,
    checkpoint_path: Path,
    model,
    val_loader,
    device: torch.device,
    ratio: float,
    threshold_metric: str,
    random_seed: int,
    complete_case_only: bool,
    zero_masked_inputs: bool,
    desc_prefix: str,
) -> Tuple[float, Dict]:
    if threshold_source == "fixed":
        return float(fixed_threshold), {
            "threshold_source": "fixed",
            "val_auc_for_threshold": np.nan,
            "val_acc_for_threshold": np.nan,
            "val_f1_for_threshold": np.nan,
            "val_num_eval_for_threshold": np.nan,
        }

    if threshold_source == "summary":
        return random_load_summary_threshold(checkpoint_path, fixed_threshold), {
            "threshold_source": "summary",
            "val_auc_for_threshold": np.nan,
            "val_acc_for_threshold": np.nan,
            "val_f1_for_threshold": np.nan,
            "val_num_eval_for_threshold": np.nan,
        }

    val_pred = random_collect_predictions_random_missing(
        model=model,
        loader=val_loader,
        device=device,
        ratio=ratio,
        random_seed=random_seed,
        complete_case_only=complete_case_only,
        zero_masked_inputs=zero_masked_inputs,
        desc=f"{desc_prefix} tune-val ratio={ratio:.2f}",
    )

    if len(val_pred["probs"]) == 0:
        return float(fixed_threshold), {
            "threshold_source": "val_ratio_empty_fallback_fixed",
            "val_auc_for_threshold": np.nan,
            "val_acc_for_threshold": np.nan,
            "val_f1_for_threshold": np.nan,
            "val_num_eval_for_threshold": int(val_pred["num_eval"]),
        }

    threshold = find_best_threshold(
        val_pred["labels"],
        val_pred["probs"],
        metric_name=threshold_metric,
    )

    val_metrics = calculate_metrics(
        val_pred["labels"],
        val_pred["probs"],
        threshold=float(threshold),
    )

    return float(threshold), {
        "threshold_source": "val_ratio",
        "val_auc_for_threshold": float(val_metrics.get("AUROC", np.nan)),
        "val_acc_for_threshold": float(val_metrics.get("Acc", np.nan)),
        "val_f1_for_threshold": float(val_metrics.get("F1", np.nan)),
        "val_num_eval_for_threshold": int(val_pred["num_eval"]),
    }


def random_evaluate_one_checkpoint(
    checkpoint_path: Path,
    config_path: Path,
    split: str,
    ratios: List[float],
    repeats: int,
    threshold_source: str,
    threshold_metric: str,
    fixed_threshold: float,
    device_override: str | None,
    batch_size_override: int | None,
    num_workers_override: int | None,
    complete_case_only: bool,
    zero_masked_inputs: bool,
    save_predictions: bool,
    pred_dir: Path,
    model_name: str,
) -> List[Dict]:
    model, cfg, device, _ = load_model_from_checkpoint(
        str(config_path),
        str(checkpoint_path),
        device=device_override,
    )

    if batch_size_override is not None:
        cfg["batch_size"] = int(batch_size_override)
    if num_workers_override is not None:
        cfg["num_workers"] = int(num_workers_override)

    loaders = _build_loaders(cfg)
    val_loader = loaders[1]
    eval_loader = random_pick_loader(split, loaders)

    seed = random_seed_from_checkpoint(checkpoint_path)
    rows: List[Dict] = []

    for ratio in ratios:
        for rep in range(int(repeats)):
            ratio_int = int(round(float(ratio) * 1000))
            base_random_seed = int(seed * 100000 + ratio_int * 100 + rep)

            threshold, threshold_info = random_determine_threshold(
                threshold_source=threshold_source,
                fixed_threshold=fixed_threshold,
                checkpoint_path=checkpoint_path,
                model=model,
                val_loader=val_loader,
                device=device,
                ratio=float(ratio),
                threshold_metric=threshold_metric,
                random_seed=base_random_seed + 17,
                complete_case_only=complete_case_only,
                zero_masked_inputs=zero_masked_inputs,
                desc_prefix=f"{model_name} seed={seed} rep={rep}",
            )

            pred = random_collect_predictions_random_missing(
                model=model,
                loader=eval_loader,
                device=device,
                ratio=float(ratio),
                random_seed=base_random_seed + 37,
                complete_case_only=complete_case_only,
                zero_masked_inputs=zero_masked_inputs,
                desc=f"{model_name} seed={seed} eval-{split} ratio={ratio:.2f} rep={rep}",
            )

            if len(pred["probs"]) == 0:
                row = {
                    "model": model_name,
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "repeat": int(rep),
                    "split": split,
                    "num_seen": int(pred["num_seen"]),
                    "num_eval": int(pred["num_eval"]),
                    "threshold": float(threshold),
                    **threshold_info,
                    "AUC": np.nan,
                    "ACC": np.nan,
                    "F1": np.nan,
                    "checkpoint": checkpoint_path.as_posix(),
                }
                rows.append(row)
                continue

            metrics = calculate_metrics(
                pred["labels"],
                pred["probs"],
                threshold=float(threshold),
            )

            row = {
                "model": model_name,
                "seed": int(seed),
                "ratio": float(ratio),
                "repeat": int(rep),
                "split": split,
                "num_seen": int(pred["num_seen"]),
                "num_eval": int(pred["num_eval"]),
                "threshold": float(threshold),
                **threshold_info,
                "AUC": float(metrics.get("AUROC", np.nan)),
                "ACC": float(metrics.get("Acc", np.nan)),
                "F1": float(metrics.get("F1", np.nan)),
                "checkpoint": checkpoint_path.as_posix(),
            }
            rows.append(row)

            if save_predictions:
                pred_df = pd.DataFrame({
                    "Participant ID": pred["pids"],
                    "label": pred["labels"].astype(int),
                    "prob": pred["probs"].astype(np.float64),
                    "pred": (pred["probs"] >= float(threshold)).astype(int),
                    "threshold": float(threshold),
                    "model": model_name,
                    "seed": int(seed),
                    "ratio": float(ratio),
                    "repeat": int(rep),
                    "split": split,
                })
                safe_model_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in model_name)
                pred_path = pred_dir / f"pred_{safe_model_name}_seed{seed}_ratio{ratio:.2f}_rep{rep}.csv"
                pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    return rows


def random_summarize(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    for col in ["ratio", "AUC", "ACC", "F1", "threshold", "num_eval"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = df.groupby(["model", "ratio"], as_index=False).agg(
        n=("F1", "count"),
        num_eval_mean=("num_eval", "mean"),
        threshold_mean=("threshold", "mean"),
        threshold_std=("threshold", "std"),
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        ACC_mean=("ACC", "mean"),
        ACC_std=("ACC", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
    )
    return summary.sort_values(["model", "ratio"]).reset_index(drop=True)


def random_write_markdown(summary_df: pd.DataFrame, out_path: Path) -> None:
    def fmt(mean_val, std_val):
        if pd.isna(mean_val):
            return ""
        if pd.isna(std_val):
            return f"{mean_val:.4f}"
        return f"{mean_val:.4f} ± {std_val:.4f}"

    lines = []
    lines.append("# Proportional Random Missing Modality Summary")
    lines.append("")
    lines.append("| Model | Missing ratio | Threshold | AUC | ACC | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        lines.append(
            "| {model} | {ratio:.2f} | {thr} | {auc} | {acc} | {f1} |".format(
                model=row["model"],
                ratio=float(row["ratio"]),
                thr=fmt(row["threshold_mean"], row["threshold_std"]),
                auc=fmt(row["AUC_mean"], row["AUC_std"]),
                acc=fmt(row["ACC_mean"], row["ACC_std"]),
                f1=fmt(row["F1_mean"], row["F1_std"]),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def random_plot_curves(summary_df: pd.DataFrame, out_dir: Path, model_name: str) -> None:
    import matplotlib.pyplot as plt

    df = summary_df[summary_df["model"] == model_name].sort_values("ratio").copy()
    if len(df) == 0:
        df = summary_df.sort_values("ratio").copy()

    x = df["ratio"].to_numpy(dtype=float) * 100.0

    for metric in ["AUC", "F1", "ACC"]:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        y = df[mean_col].to_numpy(dtype=float)
        yerr = df[std_col].fillna(0).to_numpy(dtype=float)

        plt.figure(figsize=(6.5, 4.2))
        plt.plot(x, y, marker="o", linewidth=2)
       #阴影部分 plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)
        plt.xlabel("Random missing modality ratio (%)")
        plt.ylabel(metric)
        plt.title(f"{model_name}: {metric} under proportional random missing")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        safe_model_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in model_name)
        plt.savefig(out_dir / f"random_missing_curve_{safe_model_name}_{metric.lower()}.png", dpi=300)
        plt.savefig(out_dir / f"random_missing_curve_{metric.lower()}.png", dpi=300)
        plt.close()


def random_main() -> None:
    args = random_parse_args()

    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()

    if args.out_dir is None:
        out_dir = output_root / f"{args.experiment}_random_missing_curve_{args.split}_{args.threshold_source}"
    else:
        out_dir = Path(args.out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    if args.save_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)

    ckpts = random_discover_checkpoints(output_root, args.experiment, seeds=args.seeds)
    discovered_seeds = [random_seed_from_checkpoint(p) for p in ckpts]

    print("========== Random Missing Evaluation ==========")
    print(f"config_path        : {config_path}")
    print(f"output_root        : {output_root}")
    print(f"experiment         : {args.experiment}")
    print(f"model_name         : {args.model_name}")
    print(f"split              : {args.split}")
    print(f"seeds              : {discovered_seeds}")
    print(f"ratios             : {[float(x) for x in args.ratios]}")
    print(f"repeats            : {args.repeats}")
    print(f"threshold_source   : {args.threshold_source}")
    print(f"complete_case_only : {args.complete_case_only}")
    print(f"zero_masked_inputs : {args.zero_masked_inputs}")
    print(f"out_dir            : {out_dir}")
    print("==============================================")

    all_rows: List[Dict] = []
    for ckpt in ckpts:
        rows = random_evaluate_one_checkpoint(
            checkpoint_path=ckpt,
            config_path=config_path,
            split=args.split,
            ratios=[float(x) for x in args.ratios],
            repeats=int(args.repeats),
            threshold_source=args.threshold_source,
            threshold_metric=args.threshold_metric,
            fixed_threshold=float(args.fixed_threshold),
            device_override=args.device,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            complete_case_only=bool(args.complete_case_only),
            zero_masked_inputs=bool(args.zero_masked_inputs),
            save_predictions=bool(args.save_predictions),
            pred_dir=pred_dir,
            model_name=str(args.model_name),
        )
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    summary_df = random_summarize(raw_df)

    raw_csv = out_dir / "random_missing_raw.csv"
    summary_csv = out_dir / "random_missing_summary.csv"
    summary_md = out_dir / "random_missing_summary.md"

    raw_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    random_write_markdown(summary_df, summary_md)
    random_plot_curves(summary_df, out_dir, args.model_name)

    manifest = {
        "config": str(config_path),
        "output_root": str(output_root),
        "experiment": args.experiment,
        "model_name": args.model_name,
        "split": args.split,
        "ratios": [float(x) for x in args.ratios],
        "repeats": int(args.repeats),
        "requested_seeds": None if args.seeds is None else [int(x) for x in args.seeds],
        "evaluated_seeds": [int(x) for x in discovered_seeds],
        "threshold_source": args.threshold_source,
        "threshold_metric": args.threshold_metric,
        "fixed_threshold": float(args.fixed_threshold),
        "complete_case_only": bool(args.complete_case_only),
        "zero_masked_inputs": bool(args.zero_masked_inputs),
        "num_checkpoints": len(ckpts),
        "raw_csv": raw_csv.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "summary_md": summary_md.as_posix(),
    }
    (out_dir / "random_missing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    safe_model_name = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in args.model_name)
    print("========== Done ==========")
    print(f"raw_csv    : {raw_csv}")
    print(f"summary_csv: {summary_csv}")
    print(f"summary_md : {summary_md}")
    print(f"AUC plot   : {out_dir / f'random_missing_curve_{safe_model_name}_auc.png'}")
    print(f"F1 plot    : {out_dir / f'random_missing_curve_{safe_model_name}_f1.png'}")
    print(f"ACC plot   : {out_dir / f'random_missing_curve_{safe_model_name}_acc.png'}")
    print(f"manifest   : {out_dir / 'random_missing_manifest.json'}")




def main():
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['structured', 'random'])
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.mode == 'structured':
        structured_main()
    else:
        random_main()


if __name__ == '__main__':
    main()
