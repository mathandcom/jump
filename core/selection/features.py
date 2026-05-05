from __future__ import annotations

import numpy as np
import torch

from .signals import target_onehole_signals_for_positions


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_sum": float(np.sum(arr)),
        f"{prefix}_var": float(np.var(arr)),
    }


def selected_indices(record: dict, K: int, selection_modes: list[str]) -> list[tuple[str, np.ndarray]]:
    quality = np.asarray(record["quality"], dtype=np.float64)
    prism_entropy = np.asarray(record["prism_entropy"], dtype=np.float64)
    token_frequency = np.asarray(record.get("token_frequency", np.zeros_like(quality)), dtype=np.int64)
    n_valid = int(len(quality))
    k_actual = min(int(K), n_valid)
    rng = np.random.default_rng(0)

    selections: list[tuple[str, np.ndarray]] = []
    if "random" in selection_modes:
        selections.append(("random", rng.choice(n_valid, size=k_actual, replace=False)))
    if "quality_bot" in selection_modes:
        selections.append(("quality_bot", np.argsort(quality)[:k_actual]))
    if "quality_mean_mid" in selection_modes:
        mean_quality = float(np.mean(quality))
        selections.append(("quality_mean_mid", np.argsort(np.abs(quality - mean_quality))[:k_actual]))
    if "quality_median_mid" in selection_modes:
        median_quality = float(np.median(quality))
        selections.append(("quality_median_mid", np.argsort(np.abs(quality - median_quality))[:k_actual]))
    if "rare_token" in selection_modes:
        selections.append(("rare_token", np.argsort(token_frequency, kind="stable")[:k_actual]))
    if "entropy_top" in selection_modes:
        selections.append(("entropy_top", np.argsort(prism_entropy)[-k_actual:]))
    if "entropy_bot" in selection_modes:
        selections.append(("entropy_bot", np.argsort(prism_entropy)[:k_actual]))
    return selections


def compute_features(
    prism_records: dict[str, dict],
    target_model: torch.nn.Module,
    k_values: list[int],
    selection_modes: list[str],
    mask_token_id: int,
    device: torch.device,
    hole_batch_size: int,
) -> dict[str, dict]:
    features: dict[str, dict] = {}

    for uid, record in prism_records.items():
        valid_pos = record["valid_pos"]
        quality = np.asarray(record["quality"], dtype=np.float64)
        quality_logprob = np.log(np.clip(quality, 1e-12, 1.0))
        prism_entropy = np.asarray(record["prism_entropy"], dtype=np.float64)
        prism_logit = np.asarray(record["prism_true_logit"], dtype=np.float64)
        prism_logprob = np.asarray(record["prism_true_logprob"], dtype=np.float64)

        uid_features: dict[str, float] = {}
        for K in k_values:
            for tag, idx in selected_indices(record, K, selection_modes):
                selected_pos = [valid_pos[int(i)] for i in idx]
                target = target_onehole_signals_for_positions(
                    target_model=target_model,
                    clean_ids=record["clean_ids"],
                    attn_mask=record["attn_mask"],
                    selected_pos=selected_pos,
                    mask_token_id=mask_token_id,
                    device=device,
                    hole_batch_size=hole_batch_size,
                )
                target_entropy = np.asarray(target["entropy"], dtype=np.float64)
                target_logit = np.asarray(target["true_logit"], dtype=np.float64)
                target_logprob = np.asarray(target["true_logprob"], dtype=np.float64)

                base = f"K{K}_{tag}"
                uid_features[f"{base}_k_actual"] = float(len(idx))
                uid_features.update(summarize(quality[idx], f"{base}_quality"))
                uid_features.update(summarize(quality_logprob[idx], f"{base}_prism_clean_quality_logprob"))
                uid_features.update(summarize(prism_entropy[idx], f"{base}_prism_clean_entropy"))
                uid_features.update(summarize(target_entropy, f"{base}_target_onehole_entropy"))
                uid_features.update(summarize(target_entropy - prism_entropy[idx], f"{base}_entropy_gap"))
                uid_features.update(summarize(np.abs(target_entropy - prism_entropy[idx]), f"{base}_abs_entropy_gap"))
                uid_features.update(summarize(prism_logit[idx], f"{base}_prism_clean_true_logit"))
                uid_features.update(summarize(target_logit, f"{base}_target_onehole_true_logit"))
                uid_features.update(summarize(target_logit - prism_logit[idx], f"{base}_true_logit_gap"))
                uid_features.update(summarize(prism_logprob[idx], f"{base}_prism_clean_true_logprob"))
                uid_features.update(summarize(target_logprob, f"{base}_target_onehole_true_logprob"))
                uid_features.update(summarize(target_logprob - prism_logprob[idx], f"{base}_true_logprob_gap"))
                uid_features.update(
                    summarize(
                        target_logprob - quality_logprob[idx],
                        f"{base}_target_onehole_true_logprob_minus_prism_clean_quality_logprob",
                    )
                )
                uid_features.update(
                    summarize(
                        quality_logprob[idx] - target_logprob,
                        f"{base}_prism_clean_quality_logprob_minus_target_onehole_true_logprob",
                    )
                )

        features[uid] = uid_features

    return features
