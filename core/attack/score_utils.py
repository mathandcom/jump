from __future__ import annotations

import numpy as np
import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    scores = logits.float()
    log_z = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    expected_logit = torch.sum(probs * scores, dim=-1)
    return log_z - expected_logit


def clip_suffix(clip_c: float) -> str:
    scaled10 = float(clip_c) * 10.0
    if abs(scaled10 - round(scaled10)) <= 1e-9:
        return f"c{int(round(scaled10)):02d}"
    return f"c{int(round(float(clip_c) * 1000.0)):04d}"


def compute_studentized_mean(values: np.ndarray, eps: float) -> dict[str, float | bool]:
    arr = np.asarray(values, dtype=np.float64)
    k = int(arr.size)
    mean = float(np.mean(arr)) if k > 0 else 0.0
    if k <= 1:
        sample_var = 0.0
        sample_std = 0.0
    else:
        sample_var = float(np.var(arr, ddof=1))
        sample_std = float(np.sqrt(max(sample_var, 0.0)))
    denom = float(sample_std + float(eps))
    tstat = float(np.sqrt(float(k)) * mean / denom) if denom > 0.0 else 0.0
    has_nonfinite = bool(
        (not np.all(np.isfinite(arr)))
        or (not np.isfinite(mean))
        or (not np.isfinite(sample_var))
        or (not np.isfinite(sample_std))
        or (not np.isfinite(tstat))
    )
    return {
        "mean": mean,
        "sample_var": sample_var,
        "sample_std": sample_std,
        "tstat": tstat,
        "std_near_zero": bool(sample_std <= float(eps)),
        "has_nonfinite": has_nonfinite,
    }


def compute_huber_scores(
    values: np.ndarray,
    clip_c: float,
    lower_bound: float | None,
    eps: float,
) -> dict[str, np.ndarray | float | bool]:
    arr = np.asarray(values, dtype=np.float64)
    clip_value = float(clip_c)
    clip_lower = -clip_value if lower_bound is None else float(lower_bound)
    clipped = np.clip(arr, clip_lower, clip_value)
    clipped_mask = (arr < clip_lower) | (arr > clip_value)
    denom = float(np.sqrt(np.sum(np.square(clipped))) + float(eps))
    htstat = float(np.sum(clipped) / denom) if denom > 0.0 else 0.0
    has_nonfinite = bool(
        (not np.all(np.isfinite(arr)))
        or (not np.all(np.isfinite(clipped)))
        or (not np.isfinite(denom))
        or (not np.isfinite(htstat))
    )
    return {
        "clipped": clipped,
        "hmean": float(np.mean(clipped)) if clipped.size > 0 else 0.0,
        "htstat": htstat,
        "clipped_fraction": float(np.mean(clipped_mask.astype(np.float64))) if clipped_mask.size > 0 else 0.0,
        "has_nonfinite": has_nonfinite,
    }


def compute_adaptive_huber_scores(values: np.ndarray, clip_c: float, eps: float) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    med_g = float(np.median(arr)) if arr.size > 0 else 0.0
    mad_center_raw = float(np.median(np.abs(arr - med_g))) if arr.size > 0 else 0.0
    mad_uncenter_raw = float(np.median(np.abs(arr))) if arr.size > 0 else 0.0
    mad_center = float(mad_center_raw + float(eps))
    mad_uncenter = float(mad_uncenter_raw + float(eps))

    z_center = (arr - med_g) / mad_center
    z_uncenter = arr / mad_uncenter
    clip_value = float(clip_c)
    clip_center = np.clip(z_center, -clip_value, clip_value)
    clip_uncenter = np.clip(z_uncenter, -clip_value, clip_value)
    denom_center = float(np.sqrt(np.sum(np.square(clip_center))) + float(eps))
    denom_uncenter = float(np.sqrt(np.sum(np.square(clip_uncenter))) + float(eps))

    return {
        "med_g": med_g,
        "mad_center_raw": mad_center_raw,
        "mad_uncenter_raw": mad_uncenter_raw,
        "mad_center": mad_center,
        "mad_uncenter": mad_uncenter,
        "z_center": z_center,
        "z_uncenter": z_uncenter,
        "clip_center": clip_center,
        "clip_uncenter": clip_uncenter,
        "adaphmean_madc": float(np.mean(clip_center)) if clip_center.size > 0 else 0.0,
        "adaphtstat_madc": float(np.sum(clip_center) / denom_center) if denom_center > 0.0 else 0.0,
        "adaphmean_madu": float(np.mean(clip_uncenter)) if clip_uncenter.size > 0 else 0.0,
        "adaphtstat_madu": float(np.sum(clip_uncenter) / denom_uncenter) if denom_uncenter > 0.0 else 0.0,
        "clip_fraction_madc": float(np.mean((np.abs(z_center) > clip_value).astype(np.float64))) if z_center.size > 0 else 0.0,
        "clip_fraction_madu": float(np.mean((np.abs(z_uncenter) > clip_value).astype(np.float64))) if z_uncenter.size > 0 else 0.0,
        "mad_center_near_zero": bool(mad_center_raw <= float(eps)),
        "mad_uncenter_near_zero": bool(mad_uncenter_raw <= float(eps)),
        "has_nonfinite": bool(
            (not np.all(np.isfinite(arr)))
            or (not np.all(np.isfinite(z_center)))
            or (not np.all(np.isfinite(z_uncenter)))
            or (not np.all(np.isfinite(clip_center)))
            or (not np.all(np.isfinite(clip_uncenter)))
        ),
    }
