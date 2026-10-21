from __future__ import annotations

import os
os.environ.setdefault("TORCHVISION_DISABLE_NMS_EXPORT", "1")

import re
import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _patch_torchvision_register_fake():
    original = getattr(torch.library, "register_fake", None)
    if original is None:
        return lambda: None

    def safe_register_fake(op_name):
        decorator = original(op_name)

        def wrapper(fn):
            try:
                return decorator(fn)
            except RuntimeError as exc:
                if "does not exist" in str(exc):
                    return fn
                raise

        return wrapper

    torch.library.register_fake = safe_register_fake

    def restore():
        torch.library.register_fake = original

    return restore


_restore_torchvision_patch = _patch_torchvision_register_fake()
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
if _restore_torchvision_patch is not None:
    _restore_torchvision_patch()


class TabularPreprocessor:
    """表格特征预处理"""

    def __init__(self, add_missing_indicators: bool = True, impute_strategy: str = "mean"):
        self.add_missing_indicators = bool(add_missing_indicators)
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler()
        self.feature_cols: List[str] = []
        self.output_feature_cols: List[str] = []
        self.output_dim: int = 1
        self.is_fitted = False

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Participant ID" in df.columns:
            df["Participant ID"] = df["Participant ID"].astype(str).str.strip()

        if "Sex" in df.columns:
            if df["Sex"].dtype == "O":
                df["Sex"] = (
                    df["Sex"]
                    .astype(str)
                    .str.strip()
                    .map({"M": 0, "F": 1, "Male": 0, "Female": 1})
                    .fillna(df["Sex"])
                )
            df["Sex"] = pd.to_numeric(df["Sex"], errors="coerce")

        for col in df.columns:
            if col in ("Participant ID", "labels", "has_fundus", "has_oct", "has_tabular"):
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def fit(self, df: pd.DataFrame):
        df = self._clean_dataframe(df)
        self.feature_cols = [
            c for c in df.columns
            if c not in ("Participant ID", "labels", "has_fundus", "has_oct", "has_tabular")
        ]
        if len(self.feature_cols) == 0:
            self.output_feature_cols = ["__dummy_tabular__"]
            self.output_dim = 1
            self.is_fitted = True
            return

        values_df = df[self.feature_cols]
        values = values_df.values
        self.imputer.fit(values)
        imputed = self.imputer.transform(values)
        self.scaler.fit(imputed)

        if self.add_missing_indicators:
            indicator_cols = [f"{c}__missing" for c in self.feature_cols]
            self.output_feature_cols = list(self.feature_cols) + indicator_cols
            self.output_dim = len(self.feature_cols) * 2
        else:
            self.output_feature_cols = list(self.feature_cols)
            self.output_dim = len(self.feature_cols)

        self.is_fitted = True

    def state_dict(self):
        return {
            "add_missing_indicators": self.add_missing_indicators,
            "feature_cols": list(self.feature_cols),
            "output_feature_cols": list(self.output_feature_cols),
            "output_dim": int(self.output_dim),
            "imputer_statistics": None if not hasattr(self.imputer, "statistics_") else np.asarray(self.imputer.statistics_, dtype=np.float64).tolist(),
            "scaler_mean": None if not hasattr(self.scaler, "mean_") else np.asarray(self.scaler.mean_, dtype=np.float64).tolist(),
            "scaler_scale": None if not hasattr(self.scaler, "scale_") else np.asarray(self.scaler.scale_, dtype=np.float64).tolist(),
            "is_fitted": bool(self.is_fitted),
        }

    def load_state_dict(self, state):
        self.add_missing_indicators = bool(state.get("add_missing_indicators", True))
        self.feature_cols = list(state.get("feature_cols", []))
        self.output_feature_cols = list(state.get("output_feature_cols", []))
        self.output_dim = int(state.get("output_dim", 1))
        imputer_statistics = state.get("imputer_statistics")
        scaler_mean = state.get("scaler_mean")
        scaler_scale = state.get("scaler_scale")
        if imputer_statistics is not None:
            self.imputer.statistics_ = np.asarray(imputer_statistics, dtype=np.float64)
        if scaler_mean is not None and scaler_scale is not None:
            self.scaler.mean_ = np.asarray(scaler_mean, dtype=np.float64)
            self.scaler.scale_ = np.asarray(scaler_scale, dtype=np.float64)
            self.scaler.var_ = np.square(self.scaler.scale_)
            self.scaler.n_features_in_ = len(self.scaler.mean_)
        self.is_fitted = bool(state.get("is_fitted", True))

    def transform(self, df: pd.DataFrame) -> torch.FloatTensor:
        if not self.is_fitted:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")

        df = self._clean_dataframe(df)
        if len(self.feature_cols) == 0:
            return torch.zeros(len(df), self.output_dim, dtype=torch.float32)

        values_df = df[self.feature_cols]
        missing_mask = values_df.isna().astype(np.float32).values
        values = self.imputer.transform(values_df.values)
        values = self.scaler.transform(values).astype(np.float32)
        if self.add_missing_indicators:
            values = np.concatenate([values, missing_mask], axis=1)
        return torch.tensor(values, dtype=torch.float32)


def _normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Participant ID" not in df.columns:
        raise KeyError("CSV file must contain a 'Participant ID' column.")

    def normalize_pid(x):
        s = str(x).strip()
        if s.endswith(".0"):
            s = s[:-2]
        if s.isdigit() and len(s) < 4:
            return s.zfill(4)
        return s

    df["Participant ID"] = df["Participant ID"].map(normalize_pid)
    return df


def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def build_modality_index(directory: Optional[str], suffix: str, exts: Tuple[str, ...]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    if not directory or not os.path.isdir(directory):
        return index

    for name in os.listdir(directory):
        full_path = os.path.join(directory, name)
        if not os.path.isfile(full_path):
            continue
        if not name.lower().endswith(exts):
            continue
        if suffix and suffix not in name:
            continue
        parts = name.split("_")
        if len(parts) < 2:
            continue
        pid = parts[0].strip()
        index.setdefault(pid, []).append(full_path)

    for pid, paths in index.items():
        index[pid] = sorted(paths, key=natural_key)
    return index


def choose_best_path(paths: List[str], preferred_suffixes: Tuple[str, ...]) -> str:
    basename_list = [os.path.basename(p) for p in paths]
    for suffix in preferred_suffixes:
        for base, path in zip(basename_list, paths):
            if suffix in base:
                return path
    return paths[0]


def prepare_metadata(
    labels_file: str,
    text_file: Optional[str],
    fundus_dir: Optional[str],
    oct_dir: Optional[str],
    require_fundus: bool = True,
    require_oct: bool = True,
    require_tabular: bool = True,
    allow_missing_modalities: bool = False,
):
    labels_df = _normalize_id_column(pd.read_csv(labels_file))
    labels_df = labels_df.dropna(subset=["labels"]).copy()
    labels_df["labels"] = pd.to_numeric(labels_df["labels"], errors="coerce")
    labels_df = labels_df.dropna(subset=["labels"]).copy()
    labels_df["labels"] = labels_df["labels"].astype(int)

    merged = labels_df.copy()
    if text_file:
        text_df = _normalize_id_column(pd.read_csv(text_file)).copy()
        text_df["has_tabular"] = True
        if allow_missing_modalities:
            merged = merged.merge(text_df, on="Participant ID", how="left")
        else:
            merged = merged.merge(text_df, on="Participant ID", how="inner" if require_tabular else "left")

    if "has_tabular" not in merged.columns:
        merged["has_tabular"] = False
    merged["has_tabular"] = merged["has_tabular"].fillna(False).astype(bool)

    fundus_index = build_modality_index(fundus_dir, "_21015_", (".png", ".jpg", ".jpeg"))
    oct_index = build_modality_index(oct_dir, "_21017_", (".zip",))

    fundus_ids = set(str(x).strip() for x in fundus_index.keys())
    oct_ids = set(str(x).strip() for x in oct_index.keys())

    merged["has_fundus"] = merged["Participant ID"].astype(str).isin(fundus_ids)
    merged["has_oct"] = merged["Participant ID"].astype(str).isin(oct_ids)

    if allow_missing_modalities:
        enabled_cols = []
        if require_fundus:
            enabled_cols.append("has_fundus")
        if require_oct:
            enabled_cols.append("has_oct")
        if require_tabular:
            enabled_cols.append("has_tabular")
        if enabled_cols:
            merged = merged[merged[enabled_cols].any(axis=1)].copy()
    else:
        valid_ids = set(merged["Participant ID"].astype(str).tolist())
        if require_fundus:
            valid_ids &= fundus_ids
        if require_oct:
            valid_ids &= oct_ids
        if require_tabular:
            valid_ids &= set(merged.loc[merged["has_tabular"], "Participant ID"].astype(str).tolist())
        merged = merged[merged["Participant ID"].astype(str).isin(valid_ids)].copy()

    merged = merged.sort_values("Participant ID").reset_index(drop=True)
    return merged, fundus_index, oct_index


def get_transforms(cfg: dict, mode: str = "train"):
    fundus_size = int(cfg.get("fundus_size", 384))
    oct_size = int(cfg.get("oct_size", 224))
    fundus_mean = cfg.get("fundus_mean", [0.485, 0.456, 0.406])
    fundus_std = cfg.get("fundus_std", [0.229, 0.224, 0.225])
    oct_mean = cfg.get("oct_mean", [0.5, 0.5, 0.5])
    oct_std = cfg.get("oct_std", [0.25, 0.25, 0.25])

    is_train = str(mode).lower() == "train"

    if is_train:
        fundus_transform = T.Compose([
            T.Resize((int(fundus_size * 1.08), int(fundus_size * 1.08)), interpolation=InterpolationMode.BILINEAR),
            T.RandomResizedCrop(
                fundus_size,
                scale=(0.88, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(8, interpolation=InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.05, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean=fundus_mean, std=fundus_std),
        ])
        oct_transform = T.Compose([
            T.Resize((oct_size, oct_size), interpolation=InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=oct_mean, std=oct_std),
        ])
    else:
        fundus_transform = T.Compose([
            T.Resize((fundus_size, fundus_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=fundus_mean, std=fundus_std),
        ])
        oct_transform = T.Compose([
            T.Resize((oct_size, oct_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=oct_mean, std=oct_std),
        ])

    return {"fundus": fundus_transform, "oct": oct_transform}


def _extract_target_size(transform_obj, default: int = 224) -> int:
    """安全提取最终输出尺寸：优先取最后一个有效 size，而不是第一个。"""
    if transform_obj is None:
        return int(default)

    ops = getattr(transform_obj, "transforms", [transform_obj])

    found_sizes = []
    for op in ops:
        size = getattr(op, "size", None)
        if size is None:
            continue

        if isinstance(size, int):
            found_sizes.append(int(size))
        elif isinstance(size, (tuple, list)) and len(size) > 0:
            found_sizes.append(int(size[0]))

    if not found_sizes:
        return int(default)

    return int(found_sizes[-1])


class GlaucomaDataset(Dataset):
    def __init__(
        self,
        records_df: pd.DataFrame,
        fundus_index: Dict[str, List[str]],
        oct_index: Dict[str, List[str]],
        tabular_preprocessor: TabularPreprocessor,
        transforms: dict,
        oct_slices: int = 32,
        use_fundus: bool = True,
        use_oct: bool = True,
        use_tabular: bool = True,
        is_train: bool = False,
        slice_jitter: int = 1,
        allow_missing_modalities: bool = False,
        dummy_tabular_dim: int = 1,
    ):
        self.records_df = records_df.reset_index(drop=True).copy()
        self.records_df["Participant ID"] = self.records_df["Participant ID"].astype(str)
        self.fundus_index = fundus_index
        self.oct_index = oct_index
        self.preprocessor = tabular_preprocessor
        self.transforms = transforms
        self.oct_slices = int(oct_slices)
        self.use_fundus = bool(use_fundus)
        self.use_oct = bool(use_oct)
        self.use_tabular = bool(use_tabular)
        self.is_train = bool(is_train)
        self.slice_jitter = int(slice_jitter)
        self.allow_missing_modalities = bool(allow_missing_modalities)

        self.fundus_size = _extract_target_size(transforms.get("fundus"), default=224) if self.use_fundus else 224
        self.oct_size = _extract_target_size(transforms.get("oct"), default=224) if self.use_oct else 224

        if self.use_tabular:
            self.tabular_features = self.preprocessor.transform(self.records_df)
            self.tabular_dim = int(self.tabular_features.shape[1])
        else:
            self.tabular_features = None
            pre_dim = int(getattr(self.preprocessor, "output_dim", 0))
            self.tabular_dim = max(pre_dim, int(dummy_tabular_dim))

    def __len__(self):
        return len(self.records_df)

    def _empty_fundus(self) -> torch.Tensor:
        return torch.zeros(3, self.fundus_size, self.fundus_size, dtype=torch.float32)

    def _empty_oct(self) -> torch.Tensor:
        return torch.zeros(self.oct_slices, 3, self.oct_size, self.oct_size, dtype=torch.float32)

    def _load_fundus(self, pid: str, has_fundus: bool) -> torch.Tensor:
        if not self.use_fundus or not has_fundus:
            return self._empty_fundus()

        candidates = self.fundus_index.get(pid)
        if not candidates:
            if self.allow_missing_modalities:
                return self._empty_fundus()
            raise FileNotFoundError(f"No fundus image found for Participant ID={pid}")

        path = choose_best_path(candidates, preferred_suffixes=("0_0", "1_0", "2_0"))
        img = Image.open(path).convert("RGB")
        return self.transforms["fundus"](img)

    def _sample_oct_indices(self, num_frames: int) -> np.ndarray:
        if num_frames <= 0:
            return np.zeros(self.oct_slices, dtype=int)

        centers = np.linspace(0, num_frames - 1, self.oct_slices)
        if self.is_train and self.slice_jitter > 0:
            max_jitter = max(1, num_frames // max(self.oct_slices * 4, 1))
            jitter = np.random.randint(-max_jitter, max_jitter + 1, size=self.oct_slices)
            centers = centers + jitter

        centers = np.clip(np.round(centers), 0, num_frames - 1).astype(int)
        return centers

    def _load_oct_volume(self, pid: str, has_oct: bool) -> torch.Tensor:
        if not self.use_oct or not has_oct:
            return self._empty_oct()

        candidates = self.oct_index.get(pid)
        if not candidates:
            if self.allow_missing_modalities:
                return self._empty_oct()
            raise FileNotFoundError(f"No OCT zip found for Participant ID={pid}")

        path = choose_best_path(candidates, preferred_suffixes=("0_0", "1_0", "2_0"))
        slices: List[torch.Tensor] = []

        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if m.lower().endswith((".png", ".jpg", ".jpeg")) and not m.endswith("/")]
            members = sorted(members, key=natural_key)

            if len(members) == 0:
                if self.allow_missing_modalities:
                    return self._empty_oct()
                raise RuntimeError(f"Empty OCT zip for Participant ID={pid}: {path}")

            indices = self._sample_oct_indices(len(members))
            for idx in indices:
                with zf.open(members[idx]) as fp:
                    img = Image.open(BytesIO(fp.read())).convert("RGB")
                    slices.append(self.transforms["oct"](img))

        while len(slices) < self.oct_slices:
            slices.append(slices[-1].clone())

        return torch.stack(slices[: self.oct_slices], dim=0)

    def __getitem__(self, index: int):
        row = self.records_df.iloc[index]
        pid = str(row["Participant ID"])
        label = torch.tensor(float(row["labels"]), dtype=torch.float32)

        has_fundus = bool(row["has_fundus"]) if "has_fundus" in row.index else self.use_fundus
        has_oct = bool(row["has_oct"]) if "has_oct" in row.index else self.use_oct
        has_tabular = bool(row["has_tabular"]) if "has_tabular" in row.index else self.use_tabular

        fundus = self._load_fundus(pid, has_fundus)
        oct_volume = self._load_oct_volume(pid, has_oct)

        if self.use_tabular and has_tabular and self.tabular_features is not None:
            tab = self.tabular_features[index]
        else:
            tab = torch.zeros(self.tabular_dim, dtype=torch.float32)

        modality_mask = torch.tensor([has_fundus, has_oct, has_tabular], dtype=torch.bool)

        fundus = fundus.clone().contiguous()
        oct_volume = oct_volume.clone().contiguous()
        tab = tab.clone().contiguous()
        label = label.clone().contiguous()
        modality_mask = modality_mask.clone().contiguous()

        return fundus, oct_volume, tab, label, pid, modality_mask