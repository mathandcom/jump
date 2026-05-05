from __future__ import annotations

import argparse
import hashlib

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _parse_bool(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes")


def _choose_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _stable_uid_seed(uid: str, k: int, base_seed: int) -> int:
    digest = hashlib.sha256(f"{uid}|{k}|{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.where(np.asarray(fpr) <= target_fpr)[0]
    return float(np.max(tpr[idx])) if len(idx) else 0.0


def report_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "tpr_at_fpr_10": tpr_at_fpr(labels, scores, 0.10),
        "tpr_at_fpr_1": tpr_at_fpr(labels, scores, 0.01),
        "tpr_at_fpr_0_1": tpr_at_fpr(labels, scores, 0.001),
    }


def report_metrics_with_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstraps: int = 10,
    seed: int = 42,
) -> dict:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {}

    rng = np.random.default_rng(seed)
    n = len(labels)
    metric_samples = {
        "roc_auc": [],
        "pr_auc": [],
        "tpr_at_fpr_10": [],
        "tpr_at_fpr_1": [],
        "tpr_at_fpr_0_1": [],
    }

    for _ in range(max(1, int(n_bootstraps))):
        idx = rng.choice(n, size=n, replace=True) if n > 1 else np.zeros(1, dtype=int)
        sample_labels = labels[idx]
        sample_scores = scores[idx]
        if len(np.unique(sample_labels)) < 2:
            for key in metric_samples:
                metric_samples[key].append(np.nan)
            continue
        point = report_metrics(sample_labels, sample_scores)
        for key in metric_samples:
            metric_samples[key].append(point[key])

    out: dict[str, float | str | int] = {"n_bootstrap_samples": int(n_bootstraps)}
    for key, values in metric_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_display"] = f"{mean:.4f} ± {std:.4f}"
    return out
