from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(device_name: Optional[str] = None) -> torch.device:
    preferred = (device_name or "cuda").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_logger(log_dir: str, name: str = "BaselineConcat"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    print(f"[*] 日志保存至: {log_file}")
    return logging.getLogger(name)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int, warmup_epochs: int = 0, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.max_epochs = int(max_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr = float(min_lr)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def state_dict(self):
        return {
            "max_epochs": self.max_epochs,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "base_lrs": self.base_lrs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.max_epochs = int(state_dict["max_epochs"])
        self.warmup_epochs = int(state_dict["warmup_epochs"])
        self.min_lr = float(state_dict["min_lr"])
        self.base_lrs = list(state_dict["base_lrs"])
        self.last_epoch = int(state_dict["last_epoch"])

    def get_epoch_lrs(self, epoch: int):
        lrs = []
        for base_lr in self.base_lrs:
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                scale = float(epoch + 1) / float(self.warmup_epochs)
                lr = max(self.min_lr, base_lr * scale)
            else:
                denom = max(self.max_epochs - self.warmup_epochs, 1)
                progress = float(epoch - self.warmup_epochs) / float(denom)
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(float(lr))
        return lrs

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        new_lrs = self.get_epoch_lrs(self.last_epoch)
        for lr, group in zip(new_lrs, self.optimizer.param_groups):
            group["lr"] = float(lr)


def save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_checkpoint(state: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def safe_torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def parse_override_value(raw: str):
    low = raw.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        if "." in raw or "e" in low:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def apply_cli_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    cfg = dict(cfg)
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        cfg[key] = parse_override_value(value)
    return cfg


def _to_prob(y_pred_prob):
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64)
    if np.any(y_pred_prob > 1) or np.any(y_pred_prob < 0):
        y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred_prob, -60, 60)))
    return y_pred_prob


def calculate_metrics(y_true, y_pred_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = _to_prob(y_pred_prob)
    y_pred = (y_pred_prob >= threshold).astype(int)

    try:
        auroc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auroc = 0.5

    try:
        auprc = average_precision_score(y_true, y_pred_prob)
    except ValueError:
        auprc = 0.0

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Acc": float(acc),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "BalancedAcc": float(balanced_acc),
        "Threshold": float(threshold),
    }


def find_best_threshold(y_true, y_pred_prob, metric_name: str = "F1"):
    y_true = np.asarray(y_true).astype(int)
    prob = _to_prob(y_pred_prob)
    candidate_thresholds = np.linspace(0.1, 0.9, 81)
    metric_name = str(metric_name).upper()

    best_thr = 0.5
    best_score = -1.0
    for thr in candidate_thresholds:
        metrics = calculate_metrics(y_true, prob, threshold=float(thr))
        score = metrics.get(metric_name, metrics.get("F1", 0.0))
        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)
    return best_thr


def format_report_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    key_map = {
        "AUROC": "AUC",
        "AUPRC": "AUPRC",
        "F1": "F1",
        "Acc": "ACC",
        "Sensitivity": "Sensitivity",
        "Specificity": "Specificity",
        "BalancedAcc": "BalancedAcc",
        "Threshold": "Threshold",
    }
    out = {}
    for src_key, dst_key in key_map.items():
        if src_key in metrics:
            out[dst_key] = float(metrics[src_key])
    return out


def round_nested_floats(obj: Any, digits: int = 4):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {k: round_nested_floats(v, digits=digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_nested_floats(v, digits=digits) for v in obj]
    if isinstance(obj, tuple):
        return tuple(round_nested_floats(v, digits=digits) for v in obj)
    return obj
