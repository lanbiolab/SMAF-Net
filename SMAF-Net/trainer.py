from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import GlaucomaDataset, TabularPreprocessor, get_transforms, prepare_metadata
from models import BaselineConcatModel
from utils import (
    AverageMeter,
    WarmupCosineScheduler,
    apply_cli_overrides,
    calculate_metrics,
    find_best_threshold,
    format_report_metrics,
    get_device,
    get_logger,
    round_nested_floats,
    safe_torch_load,
    save_checkpoint,
    save_json,
    seed_everything,
    seed_worker,
)


def _split_dataframe(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_file = cfg.get("split_file")

    if split_file:
        with open(split_file, "r", encoding="utf-8") as f:
            split_ids = json.load(f)

        id_to_row = {str(pid): i for i, pid in enumerate(df["Participant ID"].astype(str).tolist())}

        def pick(ids):
            rows = [id_to_row[str(pid)] for pid in ids if str(pid) in id_to_row]
            return df.iloc[rows].reset_index(drop=True)

        train_df = pick(split_ids.get("train_ids", []))
        val_df = pick(split_ids.get("val_ids", []))
        test_df = pick(split_ids.get("test_ids", []))
        return train_df, val_df, test_df

    train_ratio, val_ratio, test_ratio = cfg.get("split_ratio", [0.7, 0.15, 0.15])
    if abs(float(train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("split_ratio must sum to 1.0")

    seed = int(cfg.get("seed", 42))
    labels = df["labels"].astype(int)

    train_df, tmp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    val_portion = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=(1.0 - val_portion),
        random_state=seed,
        stratify=tmp_df["labels"].astype(int),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _build_loaders(cfg: dict):
    use_fundus = bool(cfg.get("use_fundus", True))
    use_oct = bool(cfg.get("use_oct", True))
    use_tabular = bool(cfg.get("use_tabular", True))
    allow_missing = bool(cfg.get("allow_missing_modalities", False))

    records_df, fundus_index, oct_index = prepare_metadata(
        labels_file=cfg["labels_file"],
        text_file=cfg.get("text_file"),
        fundus_dir=cfg.get("fundus_dir"),
        oct_dir=cfg.get("oct_dir"),
        require_fundus=use_fundus,
        require_oct=use_oct,
        require_tabular=use_tabular,
        allow_missing_modalities=allow_missing,
    )
    if len(records_df) == 0:
        raise RuntimeError("No valid samples found after matching labels / tabular / fundus / oct.")

    train_df, val_df, test_df = _split_dataframe(records_df, cfg)

    tabular_preprocessor = TabularPreprocessor(
        add_missing_indicators=bool(cfg.get("tabular_add_missing_indicators", True))
    )
    if use_tabular:
        fit_df = train_df
        if "has_tabular" in train_df.columns and bool(train_df["has_tabular"].any()):
            fit_df = train_df[train_df["has_tabular"]].copy()
        tabular_preprocessor.fit(fit_df)
    else:
        tabular_preprocessor.fit(pd.DataFrame({"Participant ID": ["dummy"]}))

    cfg["tabular_input_dim"] = int(tabular_preprocessor.output_dim)

    dataset_args = dict(
        fundus_index=fundus_index,
        oct_index=oct_index,
        tabular_preprocessor=tabular_preprocessor,
        oct_slices=int(cfg.get("oct_slices", 32)),
        use_fundus=use_fundus,
        use_oct=use_oct,
        use_tabular=use_tabular,
        allow_missing_modalities=allow_missing,
    )

    train_set = GlaucomaDataset(
        train_df,
        transforms=get_transforms(cfg, mode="train"),
        is_train=True,
        slice_jitter=int(cfg.get("oct_slice_jitter", 1)),
        **dataset_args,
    )
    val_set = GlaucomaDataset(
        val_df,
        transforms=get_transforms(cfg, mode="val"),
        is_train=False,
        slice_jitter=0,
        **dataset_args,
    )
    test_set = GlaucomaDataset(
        test_df,
        transforms=get_transforms(cfg, mode="test"),
        is_train=False,
        slice_jitter=0,
        **dataset_args,
    )

    batch_size = int(cfg.get("batch_size", 4))
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))

    sampler = None
    shuffle = True
    if bool(cfg.get("use_weighted_sampler", True)):
        y = train_df["labels"].astype(int).tolist()
        class_counts = np.bincount(y)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1.0 / class_counts
        sample_weights = [float(class_weights[label]) for label in y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False

    common_loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(train_set, sampler=sampler, shuffle=shuffle if sampler is None else False, **common_loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_set, shuffle=False, **common_loader_args)

    split_info = {
        "num_total": int(len(records_df)),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "num_test": int(len(test_df)),
        "train_pos": int(train_df["labels"].sum()),
        "val_pos": int(val_df["labels"].sum()),
        "test_pos": int(test_df["labels"].sum()),
        "fundus_available_total": int(
            train_df.get("has_fundus", pd.Series(dtype=bool)).sum()
            + val_df.get("has_fundus", pd.Series(dtype=bool)).sum()
            + test_df.get("has_fundus", pd.Series(dtype=bool)).sum()
        ) if "has_fundus" in records_df.columns else int(len(records_df)),
        "oct_available_total": int(
            train_df.get("has_oct", pd.Series(dtype=bool)).sum()
            + val_df.get("has_oct", pd.Series(dtype=bool)).sum()
            + test_df.get("has_oct", pd.Series(dtype=bool)).sum()
        ) if "has_oct" in records_df.columns else int(len(records_df)),
        "tabular_available_total": int(
            train_df.get("has_tabular", pd.Series(dtype=bool)).sum()
            + val_df.get("has_tabular", pd.Series(dtype=bool)).sum()
            + test_df.get("has_tabular", pd.Series(dtype=bool)).sum()
        ) if "has_tabular" in records_df.columns else int(len(records_df)),
        "missing_fundus_total": int((~records_df["has_fundus"]).sum()) if "has_fundus" in records_df.columns else 0,
        "missing_oct_total": int((~records_df["has_oct"]).sum()) if "has_oct" in records_df.columns else 0,
        "missing_tabular_total": int((~records_df["has_tabular"]).sum()) if "has_tabular" in records_df.columns else 0,
    }
    return train_loader, val_loader, test_loader, split_info, cfg, tabular_preprocessor


def _move_batch_to_device(batch, device: torch.device):
    if len(batch) != 6:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    fundus, oct_volume, tab, label, pid, modality_mask = batch
    return (
        fundus.to(device, non_blocking=True),
        oct_volume.to(device, non_blocking=True),
        tab.to(device, non_blocking=True),
        label.to(device, non_blocking=True),
        pid,
        modality_mask.to(device, non_blocking=True),
    )


def apply_modality_dropout(
    modality_mask: torch.Tensor,
    drop_prob: float = 0.3,
    force_keep_one: bool = True,
) -> torch.Tensor:
    """
    Random modality dropout for missing-modality robustness.

    modality_mask:
        shape [B, M], bool tensor.
        True means this modality is available.

    drop_prob:
        probability of dropping each available modality.

    force_keep_one:
        ensure each sample keeps at least one originally available modality.
    """
    modality_mask = modality_mask.bool()

    if drop_prob <= 0:
        return modality_mask

    if drop_prob >= 1.0 and not force_keep_one:
        return torch.zeros_like(modality_mask, dtype=torch.bool)

    device = modality_mask.device

    keep_random = torch.rand(modality_mask.shape, device=device) > float(drop_prob)
    new_mask = modality_mask & keep_random

    if force_keep_one:
        empty_rows = new_mask.sum(dim=1) == 0

        if empty_rows.any():
            empty_indices = torch.where(empty_rows)[0]

            for idx in empty_indices:
                available = torch.where(modality_mask[idx])[0]
                if available.numel() > 0:
                    rand_pos = torch.randint(
                        low=0,
                        high=available.numel(),
                        size=(1,),
                        device=device,
                    )
                    chosen_modality = available[rand_pos]
                    new_mask[idx, chosen_modality] = True

    return new_mask


def parse_modality_dropout_scenario_probs(raw) -> Dict[str, float]:
    """
    Parse scenario probabilities from config.

    Supported config examples:
    modality_dropout_scenario_probs:
      full_complete: 0.20
      no_fundus: 0.10
      no_oct: 0.10
      no_tabular: 0.20
      fundus_only: 0.15
      oct_only: 0.15
      tabular_only: 0.10
    """
    #default_probs = {
    # "full_complete": 0.20,
    #   "no_fundus": 0.10,
    #   "no_oct": 0.10,
    #   "no_tabular": 0.20,
    #   "fundus_only": 0.15,
    #   "oct_only": 0.15,
    #   "tabular_only": 0.10,
    #}
    #default_probs = {
    #    "full_complete": 0.15,
    #    "no_fundus": 0.05,
    #    "no_oct": 0.05,
    #   "no_tabular": 0.25,
    #    "fundus_only": 0.18,
    #    "oct_only": 0.22,
    #    "tabular_only": 0.10,
    #}
    default_probs = {
        "full_complete": 0.18,
        "no_fundus": 0.07,
        "no_oct": 0.07,
        "no_tabular": 0.22,
        "fundus_only": 0.20,
        "oct_only": 0.16,
        "tabular_only": 0.10,
    }

    if raw is None:
        return default_probs

    if isinstance(raw, dict):
        out = dict(default_probs)
        for k, v in raw.items():
            out[str(k)] = float(v)
        return out

    return default_probs


def apply_scenario_balanced_modality_dropout(
    modality_mask: torch.Tensor,
    scenario_probs: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Scenario-balanced modality dropout for 3 modalities in fixed dataset order:
    [fundus, oct, tabular].

    It samples a missing-modality scenario for each sample, then intersects it with
    the actually available modalities. If the sampled scenario leaves no available
    modality, it randomly keeps one originally available modality.
    """
    modality_mask = modality_mask.bool()
    device = modality_mask.device
    batch_size = modality_mask.size(0)

    if modality_mask.ndim != 2 or modality_mask.size(1) != 3:
        raise ValueError(f"Expected modality_mask shape [B, 3], got {tuple(modality_mask.shape)}")

    if scenario_probs is None:
        scenario_probs = parse_modality_dropout_scenario_probs(None)

    names = list(scenario_probs.keys())
    probs = torch.tensor([float(scenario_probs[k]) for k in names], dtype=torch.float32, device=device)
    probs = probs.clamp_min(0)
    if float(probs.sum().item()) <= 0:
        return modality_mask
    probs = probs / probs.sum()

    scenario_to_mask = {
        "full_complete": [True, True, True],
        "no_fundus": [False, True, True],
        "no_oct": [True, False, True],
        "no_tabular": [True, True, False],
        "fundus_only": [True, False, False],
        "oct_only": [False, True, False],
        "tabular_only": [False, False, True],
    }

    sampled = torch.multinomial(probs, num_samples=batch_size, replacement=True)
    new_mask = torch.zeros_like(modality_mask, dtype=torch.bool)

    for i, idx in enumerate(sampled.tolist()):
        scenario = names[idx]
        target_list = scenario_to_mask.get(scenario, scenario_to_mask["full_complete"])
        target = torch.tensor(target_list, dtype=torch.bool, device=device)

        row_mask = modality_mask[i] & target

        if row_mask.sum() == 0:
            available = torch.where(modality_mask[i])[0]
            if available.numel() > 0:
                chosen = available[torch.randint(0, available.numel(), (1,), device=device)]
                row_mask[chosen] = True

        new_mask[i] = row_mask

    return new_mask


def compute_auxiliary_losses(
    outputs: dict,
    labels: torch.Tensor,
    modality_mask: torch.Tensor,
    criterion,
    cfg: dict,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute modality-specific auxiliary losses.

    Only applies auxiliary loss to samples where the corresponding modality is
    available after modality dropout.
    """
    device = labels.device
    zero = torch.zeros((), dtype=labels.dtype, device=device)

    if not bool(cfg.get("use_aux_heads", True)):
        return zero, {}

    labels = labels.float()
    modality_mask = modality_mask.bool()

    aux_fundus_weight = float(cfg.get("aux_fundus_weight", 0.3))
    aux_oct_weight = float(cfg.get("aux_oct_weight", 0.3))
    aux_tabular_weight = float(cfg.get("aux_tabular_weight", 0.1))

    aux_loss = zero
    aux_logs: Dict[str, float] = {}

    if "fundus_logits" in outputs and aux_fundus_weight > 0:
        mask = modality_mask[:, 0]
        if mask.any():
            loss_f = criterion(outputs["fundus_logits"][mask], labels[mask])
            aux_loss = aux_loss + aux_fundus_weight * loss_f
            aux_logs["aux_fundus_loss"] = float(loss_f.detach().cpu())

    if "oct_logits" in outputs and aux_oct_weight > 0:
        mask = modality_mask[:, 1]
        if mask.any():
            loss_o = criterion(outputs["oct_logits"][mask], labels[mask])
            aux_loss = aux_loss + aux_oct_weight * loss_o
            aux_logs["aux_oct_loss"] = float(loss_o.detach().cpu())

    if "tabular_logits" in outputs and aux_tabular_weight > 0:
        mask = modality_mask[:, 2]
        if mask.any():
            loss_t = criterion(outputs["tabular_logits"][mask], labels[mask])
            aux_loss = aux_loss + aux_tabular_weight * loss_t
            aux_logs["aux_tabular_loss"] = float(loss_t.detach().cpu())

    return aux_loss, aux_logs


def _get_trainable_params(module):
    params = []
    seen = set()
    if module is None:
        return params
    for p in module.parameters():
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))
    return params


def _build_optimizer(model, cfg: dict):
    default_lr = float(cfg.get("learning_rate", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-2))

    param_groups = []
    lr_map = {
        "fundus_encoder": float(cfg.get("lr_fundus", default_lr)),
        "oct_encoder": float(cfg.get("lr_oct", default_lr)),
        "tabular_encoder": float(cfg.get("lr_tabular", default_lr)),
        "fusion": float(cfg.get("lr_fusion", default_lr)),
        "classifier": float(cfg.get("lr_classifier", default_lr)),
        "fundus_head": float(cfg.get("lr_classifier", default_lr)),
        "oct_head": float(cfg.get("lr_classifier", default_lr)),
        "tabular_head": float(cfg.get("lr_classifier", default_lr)),
    }

    for group_name in (
        "fundus_encoder",
        "oct_encoder",
        "tabular_encoder",
        "fusion",
        "classifier",
        "fundus_head",
        "oct_head",
        "tabular_head",
    ):
        module = getattr(model, group_name, None)
        if module is None:
            continue
        params = _get_trainable_params(module)
        if params:
            param_groups.append({"params": params, "lr": lr_map[group_name], "group_name": group_name})

    if not param_groups:
        raise RuntimeError("No trainable parameters found when building optimizer.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _run_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    amp_enabled: bool,
    grad_clip: float,
    modality_dropout_rate: float = 0.0,
    modality_dropout_mode: str = "random",
    modality_dropout_scenario_probs: Optional[Dict[str, float]] = None,
    cfg: Optional[dict] = None,
):
    model.train()
    meter = AverageMeter()

    use_amp = amp_enabled and device.type == "cuda"
    amp_ctx = (lambda: autocast(device_type="cuda", enabled=True)) if use_amp else nullcontext

    for batch in tqdm(loader, desc="train", leave=False):
        fundus, oct_volume, tab, label, _, modality_mask = _move_batch_to_device(batch, device)

        if modality_dropout_mode == "scenario_balanced" and float(modality_dropout_rate) > 0:
            modality_mask = apply_scenario_balanced_modality_dropout(
                modality_mask=modality_mask,
                scenario_probs=modality_dropout_scenario_probs,
            )
        else:
            modality_mask = apply_modality_dropout(
                modality_mask=modality_mask,
                drop_prob=float(modality_dropout_rate),
                force_keep_one=True,
            )

        optimizer.zero_grad(set_to_none=True)

        with amp_ctx():
            outputs = model(fundus, oct_volume, tab, modality_mask)

            # Main loss: by default supervise fusion_logits so the fusion branch remains trained
            # even when final logits are dynamically routed to a single-modality head.
            if cfg is not None and bool(cfg.get("main_loss_on_fusion_logits", True)) and "fusion_logits" in outputs:
                main_logits = outputs["fusion_logits"]
            else:
                main_logits = outputs["logits"]

            main_loss = criterion(main_logits, label.float())
            aux_loss, _ = compute_auxiliary_losses(
                outputs=outputs,
                labels=label,
                modality_mask=modality_mask,
                criterion=criterion,
                cfg=cfg or {},
            )
            loss = main_loss + aux_loss

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        meter.update(loss.item(), n=fundus.size(0))

    return meter.avg


@torch.no_grad()
def inference(model, loader, device: torch.device):
    model.eval()
    all_probs, all_labels, all_pids = [], [], []

    for batch in tqdm(loader, desc="infer", leave=False):
        fundus, oct_volume, tab, label, pid, modality_mask = _move_batch_to_device(batch, device)
        outputs = model(fundus, oct_volume, tab, modality_mask)
        probs = outputs["prob"].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(label.cpu().numpy().tolist())
        all_pids.extend(list(pid))

    return np.asarray(all_probs, dtype=np.float64), np.asarray(all_labels, dtype=np.float64), all_pids


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, threshold: Optional[float] = None, threshold_metric: str = "F1"):
    probs, labels, pids = inference(model, loader, device)
    labels = labels.astype(int)

    if threshold is None:
        threshold = find_best_threshold(labels, probs, metric_name=threshold_metric)

    metrics = calculate_metrics(labels, probs, threshold=threshold)
    pred_df = pd.DataFrame({
        "Participant ID": pids,
        "label": labels,
        "prob": probs,
        "pred": (probs >= threshold).astype(int),
    })
    return metrics, pred_df


def train_from_config(config_path: str, cli_overrides=None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_cli_overrides(cfg, cli_overrides)

    seed_everything(int(cfg.get("seed", 42)))

    output_dir = cfg.get("output_dir", "outputs/baseline_concat")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = get_logger(log_dir)
    device = get_device(cfg.get("device", "cuda"))
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, split_info, cfg, tabular_preprocessor = _build_loaders(cfg)
    save_json(split_info, os.path.join(output_dir, "split_info.json"))
    logger.info(f"Split info: {split_info}")

    model = BaselineConcatModel(cfg).to(device)

    optimizer = _build_optimizer(model, cfg)
    for i, group in enumerate(optimizer.param_groups):
        logger.info(
            f"Optimizer group[{i}] name={group.get('group_name', f'group_{i}')} "
            f"lr={group['lr']:.8f} num_params={sum(p.numel() for p in group['params'])}"
        )

    scheduler = WarmupCosineScheduler(
        optimizer,
        max_epochs=int(cfg.get("epochs", 30)),
        warmup_epochs=int(cfg.get("warmup_epochs", 3)),
        min_lr=float(cfg.get("min_learning_rate", 1e-6)),
    )

    train_labels = train_loader.dataset.records_df["labels"].astype(int).to_numpy()
    pos_count = max(int(train_labels.sum()), 1)
    neg_count = max(int(len(train_labels) - train_labels.sum()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler(enabled=bool(cfg.get("amp", True)) and device.type == "cuda")

    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0

    patience = int(cfg.get("patience", 8))
    epochs = int(cfg.get("epochs", 30))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    selection_metric = str(cfg.get("selection_metric", "AUROC"))
    threshold_metric = str(cfg.get("threshold_metric", "F1"))
    log_eval_threshold = float(cfg.get("log_eval_threshold", 0.5))

    modality_dropout_rate = float(cfg.get("modality_dropout_rate", 0.0))
    modality_dropout_mode = str(cfg.get("modality_dropout_mode", "random"))
    modality_dropout_scenario_probs = parse_modality_dropout_scenario_probs(
        cfg.get("modality_dropout_scenario_probs")
    )
    logger.info(f"Modality dropout rate: {modality_dropout_rate:.4f}")
    logger.info(f"Modality dropout mode: {modality_dropout_mode}")
    if modality_dropout_mode == "scenario_balanced":
        logger.info(f"Modality dropout scenario probs: {modality_dropout_scenario_probs}")
    logger.info(
        "Aux heads: use_aux_heads={use_aux} weights fundus={wf:.3f} oct={wo:.3f} tabular={wt:.3f} "
        "final_logit_mode={mode}".format(
            use_aux=bool(cfg.get("use_aux_heads", True)),
            wf=float(cfg.get("aux_fundus_weight", 0.3)),
            wo=float(cfg.get("aux_oct_weight", 0.3)),
            wt=float(cfg.get("aux_tabular_weight", 0.1)),
            mode=str(cfg.get("final_logit_mode", "single_when_one")),
        )
    )

    history = []
    for epoch in range(epochs):
        scheduler.step(epoch)
        current_lrs = {
            group.get("group_name", f"group_{i}"): round(float(group["lr"]), 8)
            for i, group in enumerate(optimizer.param_groups)
        }

        train_loss = _run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=bool(cfg.get("amp", True)),
            grad_clip=grad_clip,
            modality_dropout_rate=modality_dropout_rate,
            modality_dropout_mode=modality_dropout_mode,
            modality_dropout_scenario_probs=modality_dropout_scenario_probs,
            cfg=cfg,
        )

        val_metrics_log, val_pred_df_log = evaluate_model(
            model=model,
            loader=val_loader,
            device=device,
            threshold=log_eval_threshold,
            threshold_metric=threshold_metric,
        )

        metric_value = float(val_metrics_log.get(selection_metric, val_metrics_log.get("AUROC", 0.0)))
        report_val_metrics_log = format_report_metrics(val_metrics_log)

        record = {
            "epoch": epoch + 1,
            "lrs": current_lrs,
            "train_loss": round(float(train_loss), 4),
            "log_eval_threshold": round(float(log_eval_threshold), 4),
            "modality_dropout_rate": round(float(modality_dropout_rate), 4),
            **{f"val_{k}": float(v) for k, v in report_val_metrics_log.items()},
        }
        history.append(record)
        save_json(round_nested_floats({"history": history}, digits=4), os.path.join(output_dir, "history.json"))

        logger.info(
            f"Epoch [{epoch + 1}/{epochs}] lrs={current_lrs} "
            f"loss={train_loss:.4f} val_AUC={report_val_metrics_log['AUC']:.4f} "
            f"val_ACC={report_val_metrics_log['ACC']:.4f} val_F1={report_val_metrics_log['F1']:.4f}"
        )

        ckpt = {
            "epoch": epoch + 1,
            "config": cfg,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics_log_threshold": val_metrics_log,
            "log_eval_threshold": float(log_eval_threshold),
            "modality_dropout_rate": float(modality_dropout_rate),
            "modality_dropout_mode": modality_dropout_mode,
            "modality_dropout_scenario_probs": modality_dropout_scenario_probs,
            "tabular_preprocessor_state": tabular_preprocessor.state_dict(),
        }
        save_checkpoint(ckpt, os.path.join(checkpoint_dir, "last.ckpt"))

        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch + 1
            bad_epochs = 0
            save_checkpoint(ckpt, os.path.join(checkpoint_dir, "best.ckpt"))
            val_pred_df_log.to_csv(os.path.join(output_dir, "val_predictions_best_log_threshold.csv"), index=False)
            save_json(format_report_metrics(val_metrics_log), os.path.join(output_dir, "val_metrics_log_threshold.json"))
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    best_ckpt = safe_torch_load(os.path.join(checkpoint_dir, "best.ckpt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    val_metrics_tuned, val_pred_df_tuned = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
        threshold=None,
        threshold_metric=threshold_metric,
    )
    best_threshold = float(val_metrics_tuned["Threshold"])

    test_metrics, test_pred_df = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        threshold=best_threshold,
        threshold_metric=threshold_metric,
    )

    report_val_metrics_tuned = format_report_metrics(val_metrics_tuned)
    report_test_metrics = format_report_metrics(test_metrics)
    save_json(report_val_metrics_tuned, os.path.join(output_dir, "val_metrics.json"))
    save_json(report_test_metrics, os.path.join(output_dir, "test_metrics.json"))

    if bool(cfg.get("save_predictions", True)):
        val_pred_df_tuned.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
        test_pred_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)

    summary = {
        "best_epoch": int(best_epoch),
        "selection_metric": selection_metric,
        "best_selection_metric_value": round(float(best_metric), 4),
        "log_eval_threshold": round(float(log_eval_threshold), 4),
        "best_threshold_from_val": round(float(best_threshold), 4),
        "modality_dropout_rate": round(float(modality_dropout_rate), 4),
        "modality_dropout_mode": modality_dropout_mode,
        "modality_dropout_scenario_probs": modality_dropout_scenario_probs,
        "use_aux_heads": bool(cfg.get("use_aux_heads", True)),
        "aux_fundus_weight": float(cfg.get("aux_fundus_weight", 0.3)),
        "aux_oct_weight": float(cfg.get("aux_oct_weight", 0.3)),
        "aux_tabular_weight": float(cfg.get("aux_tabular_weight", 0.1)),
        "final_logit_mode": str(cfg.get("final_logit_mode", "single_when_one")),
        "val_metrics": report_val_metrics_tuned,
        "test_metrics": report_test_metrics,
        "checkpoint": os.path.join(checkpoint_dir, "best.ckpt"),
    }
    save_json(round_nested_floats(summary, digits=4), os.path.join(output_dir, "train_summary.json"))

    logger.info(
        f"Training finished. Best epoch={best_epoch}, best_val_threshold={best_threshold:.4f}, "
        f"modality_dropout_rate={modality_dropout_rate:.4f}, "
        f"modality_dropout_mode={modality_dropout_mode}, "
        f"test AUC={report_test_metrics['AUC']:.4f}, test ACC={report_test_metrics['ACC']:.4f}, "
        f"test F1={report_test_metrics['F1']:.4f}"
    )
    return summary


def load_model_from_checkpoint(config_path: str, checkpoint_path: str, device: Optional[str] = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = safe_torch_load(checkpoint_path, map_location="cpu")
    # Preserve checkpoint architecture while allowing portable data/runtime paths.
    runtime_keys = (
        "labels_file", "text_file", "fundus_dir", "oct_dir", "split_file",
        "output_dir", "device", "batch_size", "num_workers", "pin_memory",
    )
    runtime = {key: cfg[key] for key in runtime_keys if key in cfg}
    cfg.update(ckpt.get("config", {}))
    cfg.update(runtime)

    dev = get_device(device or cfg.get("device", "cuda"))
    model = BaselineConcatModel(cfg).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, dev, ckpt


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train SMAF-Net or a configured fusion baseline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", nargs="*", default=None, help="Config overrides in KEY=VALUE format.")
    args = parser.parse_args()
    train_from_config(args.config, cli_overrides=args.set)


if __name__ == "__main__":
    main()
