from __future__ import annotations

import numpy as np

from jump.core.attack.score_utils import (
    clip_suffix,
    compute_adaptive_huber_scores,
    compute_huber_scores,
    compute_studentized_mean,
)
from jump.core.selection.features import summarize


def build_features(
    prism_records: dict[str, dict],
    target_maps: dict[str, dict[str, np.ndarray]],
    reference_maps: dict[str, dict[str, np.ndarray]],
    selected_k: int,
    selection_mode: str,
    group_size: int,
    prefix_counts: list[int],
    fixed_huber_clip_values: list[float],
    fixed_huber_lower_bound: float | None,
    adaptive_huber_clip_values: list[float],
    adaptive_huber_eps: float,
    tstat_eps: float,
    debug_dump_limit: int,
    hier_top_groups: list[int],
    hier_refine_modes: list[str],
    hier_target_maps: dict[str, dict[str, np.ndarray]] | None = None,
    hier_reference_maps: dict[str, dict[str, np.ndarray]] | None = None,
    save_selector_diagnostics: bool = False,
) -> tuple[dict[str, dict], list[dict], dict[str, object], dict[str, dict]]:
    features: dict[str, dict] = {}
    adaptive_debug: list[dict] = []
    selector_diagnostics: dict[str, dict] = {}
    sanity: dict[str, object] = {
        "p32_sequence_count": 0,
        "p32_exact_32_count": 0,
        "tstat_std_near_zero_count": 0,
        "tstat_nonfinite_count": 0,
        "raw_huber_nonfinite_count": 0,
        "adaptive_huber_application_count": 0,
        "adaptive_huber_nonfinite_count": 0,
        "adaptive_huber_madc_mad_near_zero_count": 0,
        "adaptive_huber_madu_mad_near_zero_count": 0,
        "adaptive_huber_bad_score_count": 0,
        "adaptive_huber_clip_stats": {},
    }
    clip_stats: dict[str, dict[str, float]] = {
        f"c{int(round(float(c) * 10)):02d}": {
            "count": 0.0,
            "madc_clipped_fraction_sum": 0.0,
            "madu_clipped_fraction_sum": 0.0,
        }
        for c in adaptive_huber_clip_values
    }
    base = f"K{int(selected_k)}_{selection_mode}"
    shared_keys = sorted(set(target_maps) & set(reference_maps))
    grouped_keys: dict[str, list[str]] = {}
    for key in shared_keys:
        uid = target_maps[key]["uid"]
        grouped_keys.setdefault(uid, []).append(key)

    for uid, keys in grouped_keys.items():
        record = prism_records[uid]
        quality = np.asarray(record["quality"], dtype=np.float64)
        prism_entropy = np.asarray(record["prism_entropy"], dtype=np.float64)
        uid_features: dict[str, float] = {}
        group_prob_gap_means: list[float] = []
        group_prob_gap_maxs: list[float] = []

        for key in sorted(keys):
            target = target_maps[key]
            reference = reference_maps[key]
            idx = np.asarray(target["idx"], dtype=np.int64)
            if not np.array_equal(idx, np.asarray(reference["idx"], dtype=np.int64)):
                raise RuntimeError(f"Selection mismatch for {uid} {key}")

            group_tag = str(target["group_tag"])
            group_base = f"{base}_{group_tag}"
            tgt_prob = np.asarray(target["true_prob"], dtype=np.float64)
            ref_prob = np.asarray(reference["true_prob"], dtype=np.float64)
            tgt_logit = np.asarray(target["true_logit"], dtype=np.float64)
            ref_logit = np.asarray(reference["true_logit"], dtype=np.float64)
            tgt_logprob = np.asarray(target["true_logprob"], dtype=np.float64)
            ref_logprob = np.asarray(reference["true_logprob"], dtype=np.float64)
            tgt_entropy = np.asarray(target["entropy"], dtype=np.float64)
            ref_entropy = np.asarray(reference["entropy"], dtype=np.float64)
            prob_gap = tgt_prob - ref_prob

            uid_features[f"{group_base}_k_actual"] = float(len(idx))
            uid_features.update(summarize(quality[idx], f"{group_base}_quality"))
            uid_features.update(summarize(prism_entropy[idx], f"{group_base}_prism_clean_entropy"))
            uid_features.update(summarize(tgt_prob, f"{group_base}_target_multimask_true_prob"))
            uid_features.update(summarize(ref_prob, f"{group_base}_reference_multimask_true_prob"))
            uid_features.update(summarize(prob_gap, f"{group_base}_target_ref_multimask_true_prob_gap"))
            uid_features.update(summarize(ref_prob - tgt_prob, f"{group_base}_reference_target_multimask_true_prob_gap"))
            uid_features.update(summarize(tgt_logit, f"{group_base}_target_multimask_true_logit"))
            uid_features.update(summarize(ref_logit, f"{group_base}_reference_multimask_true_logit"))
            uid_features.update(summarize(tgt_logit - ref_logit, f"{group_base}_target_ref_multimask_true_logit_gap"))
            uid_features.update(summarize(tgt_logprob, f"{group_base}_target_multimask_true_logprob"))
            uid_features.update(summarize(ref_logprob, f"{group_base}_reference_multimask_true_logprob"))
            uid_features.update(summarize(tgt_logprob - ref_logprob, f"{group_base}_target_ref_multimask_true_logprob_gap"))
            uid_features.update(summarize(ref_logprob - tgt_logprob, f"{group_base}_reference_target_multimask_true_logprob_gap"))
            uid_features.update(
                summarize(
                    np.abs(ref_logprob - quality[idx]),
                    f"{group_base}_reference_multimask_true_logprob_quality_abs_gap",
                )
            )
            uid_features.update(
                summarize(
                    np.exp(np.clip(tgt_logprob - ref_logprob, -80.0, 80.0)),
                    f"{group_base}_target_ref_multimask_true_prob_ratio",
                )
            )
            uid_features.update(summarize(tgt_entropy, f"{group_base}_target_multimask_entropy"))
            uid_features.update(summarize(ref_entropy, f"{group_base}_reference_multimask_entropy"))
            uid_features.update(summarize(tgt_entropy - ref_entropy, f"{group_base}_target_ref_multimask_entropy_gap"))
            group_prob_gap_means.append(float(np.mean(prob_gap)))
            group_prob_gap_maxs.append(float(np.max(prob_gap)))

            if group_tag == "all":
                token_gap = tgt_logprob - ref_logprob
                target_logprob = tgt_logprob
                reference_quality_abs_gap = np.abs(ref_logprob - quality[idx])
                k_tag = f"p{int(selected_k)}"
                sanity["p32_sequence_count"] = int(sanity["p32_sequence_count"]) + 1
                if int(len(idx)) == int(selected_k):
                    sanity["p32_exact_32_count"] = int(sanity["p32_exact_32_count"]) + 1

                uid_features[f"prism_{k_tag}_logprob_gap_mean"] = float(np.mean(token_gap))
                uid_features[f"prism_{k_tag}_target_multimask_true_logprob_mean"] = float(np.mean(target_logprob))
                uid_features[f"prism_{k_tag}_reference_quality_abs_gap_mean"] = float(np.mean(reference_quality_abs_gap))
                tstat_stats = compute_studentized_mean(token_gap, eps=float(tstat_eps))
                uid_features[f"prism_{k_tag}_logprob_gap_tstat"] = float(tstat_stats["tstat"])
                if bool(tstat_stats["std_near_zero"]):
                    sanity["tstat_std_near_zero_count"] = int(sanity["tstat_std_near_zero_count"]) + 1
                if bool(tstat_stats["has_nonfinite"]):
                    sanity["tstat_nonfinite_count"] = int(sanity["tstat_nonfinite_count"]) + 1

                for clip_c in fixed_huber_clip_values:
                    fixed_huber = compute_huber_scores(
                        token_gap,
                        clip_c=float(clip_c),
                        lower_bound=fixed_huber_lower_bound,
                        eps=float(adaptive_huber_eps),
                    )
                    suffix = clip_suffix(float(clip_c))
                    uid_features[f"prism_{k_tag}_logprob_gap_hmean_{suffix}"] = float(fixed_huber["hmean"])
                    uid_features[f"prism_{k_tag}_logprob_gap_htstat_{suffix}"] = float(fixed_huber["htstat"])
                    if bool(fixed_huber["has_nonfinite"]):
                        sanity["raw_huber_nonfinite_count"] = int(sanity["raw_huber_nonfinite_count"]) + 1

                    target_fixed_huber = compute_huber_scores(
                        target_logprob,
                        clip_c=float(clip_c),
                        lower_bound=fixed_huber_lower_bound,
                        eps=float(adaptive_huber_eps),
                    )
                    uid_features[f"prism_{k_tag}_target_multimask_true_logprob_hmean_{suffix}"] = float(target_fixed_huber["hmean"])
                    uid_features[f"prism_{k_tag}_target_multimask_true_logprob_htstat_{suffix}"] = float(target_fixed_huber["htstat"])
                    if bool(target_fixed_huber["has_nonfinite"]):
                        sanity["raw_huber_nonfinite_count"] = int(sanity["raw_huber_nonfinite_count"]) + 1

                    quality_abs_gap_fixed_huber = compute_huber_scores(
                        reference_quality_abs_gap,
                        clip_c=float(clip_c),
                        lower_bound=fixed_huber_lower_bound,
                        eps=float(adaptive_huber_eps),
                    )
                    uid_features[f"prism_{k_tag}_reference_quality_abs_gap_hmean_{suffix}"] = float(quality_abs_gap_fixed_huber["hmean"])
                    uid_features[f"prism_{k_tag}_reference_quality_abs_gap_htstat_{suffix}"] = float(quality_abs_gap_fixed_huber["htstat"])
                    if bool(quality_abs_gap_fixed_huber["has_nonfinite"]):
                        sanity["raw_huber_nonfinite_count"] = int(sanity["raw_huber_nonfinite_count"]) + 1

                if save_selector_diagnostics:
                    selected_token_frequency = np.asarray(
                        record.get("token_frequency", np.zeros_like(quality)),
                        dtype=np.float64,
                    )[idx]
                    diag_entry = {
                        "selected_indices": idx.tolist(),
                        "selected_positions": np.asarray(target.get("selected_positions", np.asarray([], dtype=np.int64)), dtype=np.int64).tolist(),
                        "selected_token_ids": np.asarray(target.get("selected_token_ids", np.asarray([], dtype=np.int64)), dtype=np.int64).tolist(),
                        "selector_quality": quality[idx].tolist(),
                        "selector_entropy": prism_entropy[idx].tolist(),
                        "selected_token_frequency": selected_token_frequency.tolist(),
                        "target_true_logprob": tgt_logprob.tolist(),
                        "reference_true_logprob": ref_logprob.tolist(),
                        "reference_quality_abs_gap": reference_quality_abs_gap.tolist(),
                        "token_level_gap": token_gap.tolist(),
                        "clipped_gap_by_suffix": {},
                    }
                    for clip_c in fixed_huber_clip_values:
                        clip_stats_row = compute_huber_scores(
                            token_gap,
                            clip_c=float(clip_c),
                            lower_bound=fixed_huber_lower_bound,
                            eps=float(adaptive_huber_eps),
                        )
                        diag_entry["clipped_gap_by_suffix"][clip_suffix(float(clip_c))] = np.asarray(
                            clip_stats_row["clipped"],
                            dtype=np.float64,
                        ).tolist()
                    selector_diagnostics[uid] = diag_entry

                debug_row = None
                if len(adaptive_debug) < int(debug_dump_limit):
                    debug_row = {"uid": uid, "group_tag": group_tag, "selector_indices": idx.tolist(), "token_gap": token_gap.tolist(), "adaptive": {}}

                for clip_c in adaptive_huber_clip_values:
                    stats = compute_adaptive_huber_scores(token_gap, clip_c=float(clip_c), eps=float(adaptive_huber_eps))
                    suffix = f"c{int(round(float(clip_c) * 10)):02d}"
                    uid_features[f"prism_{k_tag}_adaphmean_madc_{suffix}"] = float(stats["adaphmean_madc"])
                    uid_features[f"prism_{k_tag}_adaphtstat_madc_{suffix}"] = float(stats["adaphtstat_madc"])
                    uid_features[f"prism_{k_tag}_adaphmean_madu_{suffix}"] = float(stats["adaphmean_madu"])
                    uid_features[f"prism_{k_tag}_adaphtstat_madu_{suffix}"] = float(stats["adaphtstat_madu"])
                    sanity["adaptive_huber_application_count"] = int(sanity["adaptive_huber_application_count"]) + 1
                    if bool(stats["mad_center_near_zero"]):
                        sanity["adaptive_huber_madc_mad_near_zero_count"] = int(sanity["adaptive_huber_madc_mad_near_zero_count"]) + 1
                    if bool(stats["mad_uncenter_near_zero"]):
                        sanity["adaptive_huber_madu_mad_near_zero_count"] = int(sanity["adaptive_huber_madu_mad_near_zero_count"]) + 1
                    if bool(stats["has_nonfinite"]):
                        sanity["adaptive_huber_nonfinite_count"] = int(sanity["adaptive_huber_nonfinite_count"]) + 1
                    if not all(np.isfinite([float(stats["adaphmean_madc"]), float(stats["adaphtstat_madc"]), float(stats["adaphmean_madu"]), float(stats["adaphtstat_madu"])])):
                        sanity["adaptive_huber_bad_score_count"] = int(sanity["adaptive_huber_bad_score_count"]) + 1
                    clip_bucket = clip_stats[suffix]
                    clip_bucket["count"] += 1.0
                    clip_bucket["madc_clipped_fraction_sum"] += float(stats["clip_fraction_madc"])
                    clip_bucket["madu_clipped_fraction_sum"] += float(stats["clip_fraction_madu"])
                    if debug_row is not None:
                        debug_row["adaptive"][suffix] = {
                            "med_g": float(stats["med_g"]),
                            "mad_center_raw": float(stats["mad_center_raw"]),
                            "mad_uncenter_raw": float(stats["mad_uncenter_raw"]),
                            "z_center": np.asarray(stats["z_center"], dtype=np.float64).tolist(),
                            "z_uncenter": np.asarray(stats["z_uncenter"], dtype=np.float64).tolist(),
                            "clip_center": np.asarray(stats["clip_center"], dtype=np.float64).tolist(),
                            "clip_uncenter": np.asarray(stats["clip_uncenter"], dtype=np.float64).tolist(),
                            "clip_fraction_madc": float(stats["clip_fraction_madc"]),
                            "clip_fraction_madu": float(stats["clip_fraction_madu"]),
                            "adaphmean_madc": float(stats["adaphmean_madc"]),
                            "adaphtstat_madc": float(stats["adaphtstat_madc"]),
                            "adaphmean_madu": float(stats["adaphmean_madu"]),
                            "adaphtstat_madu": float(stats["adaphtstat_madu"]),
                        }
                if debug_row is not None:
                    adaptive_debug.append(debug_row)

        if prefix_counts:
            uid_features[f"{base}_prefix_count_count"] = float(len(group_prob_gap_means))
        elif int(group_size) > 0 and group_prob_gap_means:
            uid_features[f"{base}_group_size"] = float(int(group_size))
            uid_features[f"{base}_group_count"] = float(len(group_prob_gap_means))
            uid_features[f"{base}_grouped_target_ref_multimask_true_prob_gap_mean"] = float(np.mean(group_prob_gap_means))
            uid_features[f"{base}_grouped_target_ref_multimask_true_prob_gap_max"] = float(np.max(group_prob_gap_means))
            uid_features[f"{base}_grouped_target_ref_multimask_true_prob_gap_top2_mean"] = float(np.mean(sorted(group_prob_gap_means, reverse=True)[:2]))
            uid_features[f"{base}_grouped_target_ref_multimask_true_prob_gap_bottom2_mean"] = float(np.mean(sorted(group_prob_gap_means)[:2]))
            uid_features[f"{base}_grouped_target_ref_multimask_true_prob_gap_groupmax_mean"] = float(np.mean(group_prob_gap_maxs))
        features[uid] = uid_features

    if hier_top_groups and hier_refine_modes and hier_target_maps and hier_reference_maps:
        from jump.core.attack.hierarchy import stage1_group_rankings

        stage1_rankings = stage1_group_rankings(target_maps, reference_maps)
        hier_grouped_keys: dict[str, dict[tuple[int, str], list[str]]] = {}
        for key in sorted(set(hier_target_maps) & set(hier_reference_maps)):
            uid = str(hier_target_maps[key]["uid"])
            raw_tag = str(hier_target_maps[key]["group_tag"])
            if not raw_tag.startswith("h|"):
                continue
            parts = raw_tag.split("|")
            if len(parts) != 5:
                continue
            _, top_tag, refine_mode, _, _ = parts
            top_n = int(top_tag[1:])
            hier_grouped_keys.setdefault(uid, {}).setdefault((top_n, refine_mode), []).append(key)

        for uid, by_config in hier_grouped_keys.items():
            uid_features = features.setdefault(uid, {})
            ranked_groups = stage1_rankings.get(uid, [])
            for top_n in hier_top_groups:
                chosen_groups = ranked_groups[:min(int(top_n), len(ranked_groups))]
                if chosen_groups:
                    chosen_scores = np.asarray([row["score"] for row in chosen_groups], dtype=np.float64)
                    uid_features.update(summarize(chosen_scores, f"{base}_hier_top{int(top_n)}_stage1_target_ref_multimask_true_prob_gap"))
                    uid_features[f"{base}_hier_top{int(top_n)}_selected_group_count"] = float(len(chosen_groups))
            for (top_n, refine_mode), keys in by_config.items():
                tgt_prob_all: list[np.ndarray] = []
                ref_prob_all: list[np.ndarray] = []
                tgt_logprob_all: list[np.ndarray] = []
                ref_logprob_all: list[np.ndarray] = []
                tgt_entropy_all: list[np.ndarray] = []
                ref_entropy_all: list[np.ndarray] = []
                for key in sorted(keys):
                    target = hier_target_maps[key]
                    reference = hier_reference_maps[key]
                    tgt_prob_all.append(np.asarray(target["true_prob"], dtype=np.float64))
                    ref_prob_all.append(np.asarray(reference["true_prob"], dtype=np.float64))
                    tgt_logprob_all.append(np.asarray(target["true_logprob"], dtype=np.float64))
                    ref_logprob_all.append(np.asarray(reference["true_logprob"], dtype=np.float64))
                    tgt_entropy_all.append(np.asarray(target["entropy"], dtype=np.float64))
                    ref_entropy_all.append(np.asarray(reference["entropy"], dtype=np.float64))

                tgt_prob = np.concatenate(tgt_prob_all) if tgt_prob_all else np.array([], dtype=np.float64)
                ref_prob = np.concatenate(ref_prob_all) if ref_prob_all else np.array([], dtype=np.float64)
                tgt_logprob = np.concatenate(tgt_logprob_all) if tgt_logprob_all else np.array([], dtype=np.float64)
                ref_logprob = np.concatenate(ref_logprob_all) if ref_logprob_all else np.array([], dtype=np.float64)
                tgt_entropy = np.concatenate(tgt_entropy_all) if tgt_entropy_all else np.array([], dtype=np.float64)
                ref_entropy = np.concatenate(ref_entropy_all) if ref_entropy_all else np.array([], dtype=np.float64)
                if tgt_prob.size == 0:
                    continue

                hier_base = f"{base}_hier_top{int(top_n)}_{refine_mode}"
                uid_features[f"{hier_base}_n_masks"] = float(tgt_prob.size)
                uid_features.update(summarize(tgt_prob, f"{hier_base}_target_true_prob"))
                uid_features.update(summarize(ref_prob, f"{hier_base}_reference_true_prob"))
                uid_features.update(summarize(tgt_prob - ref_prob, f"{hier_base}_target_ref_true_prob_gap"))
                uid_features.update(summarize(ref_prob - tgt_prob, f"{hier_base}_reference_target_true_prob_gap"))
                uid_features.update(summarize(tgt_logprob, f"{hier_base}_target_true_logprob"))
                uid_features.update(summarize(ref_logprob, f"{hier_base}_reference_true_logprob"))
                uid_features.update(summarize(tgt_logprob - ref_logprob, f"{hier_base}_target_ref_true_logprob_gap"))
                uid_features.update(summarize(ref_logprob - tgt_logprob, f"{hier_base}_reference_target_true_logprob_gap"))
                uid_features.update(summarize(tgt_entropy, f"{hier_base}_target_entropy"))
                uid_features.update(summarize(ref_entropy, f"{hier_base}_reference_entropy"))
                uid_features.update(summarize(tgt_entropy - ref_entropy, f"{hier_base}_target_ref_entropy_gap"))

    for key, bucket in clip_stats.items():
        count = float(bucket["count"])
        sanity["adaptive_huber_clip_stats"][key] = {
            "count": int(count),
            "mean_clipped_fraction_madc": float(bucket["madc_clipped_fraction_sum"] / count) if count > 0.0 else 0.0,
            "mean_clipped_fraction_madu": float(bucket["madu_clipped_fraction_sum"] / count) if count > 0.0 else 0.0,
        }
    return features, adaptive_debug, sanity, selector_diagnostics
