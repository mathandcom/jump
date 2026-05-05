from __future__ import annotations

import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from jump.core.attack.args import load_token_frequency_table, parse_args
from jump.core.attack.features import build_features
from jump.core.attack.hierarchy import build_hierarchical_refine_records, stage1_group_rankings
from jump.core.attack.masking import (
    collect_target_records,
    compute_maps_for_ordered_records,
    compute_multimask_maps,
    load_reference_backbone,
    transfer_selector_indices_to_target,
)
from jump.core.shared.io import load_manifest
from jump.core.shared.metrics import _choose_dtype, _parse_csv_list, report_metrics_with_bootstrap, tpr_at_fpr
from jump.core.shared.models import (
    load_c4_prism,
    load_raw_backbone,
)
from jump.core.prism_utils import find_mask_token_id
from jump.core.selection.features import selected_indices
from jump.core.selection.signals import collect_clean_prism_signals


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()
    target_model_path = args.target_model_path or args.model_path
    target_tokenizer_path = args.target_tokenizer_path or args.tokenizer_path or target_model_path
    only_features = set(_parse_csv_list(args.only_features)) if args.only_features else set()
    prefix_counts = [int(v) for v in _parse_csv_list(args.prefix_counts)] if args.prefix_counts else []
    fixed_huber_clip_values = [float(v) for v in _parse_csv_list(args.fixed_huber_clip_values)] if str(args.fixed_huber_clip_values).strip().lower() not in {"", "none"} else []
    adaptive_huber_clip_values = [float(v) for v in _parse_csv_list(args.adaptive_huber_clip_values)] if str(args.adaptive_huber_clip_values).strip().lower() not in {"", "none"} else []
    hier_top_groups = [int(v) for v in _parse_csv_list(args.hier_top_groups)] if args.hier_top_groups else []
    hier_refine_modes = _parse_csv_list(args.hier_refine_modes) if args.hier_refine_modes else []
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _choose_dtype(args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.tokenizer_path or args.model_path
    selector_tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if selector_tokenizer.pad_token_id is None:
        selector_tokenizer.pad_token_id = selector_tokenizer.eos_token_id or 0
    selector_model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    selector_special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
        value = getattr(selector_tokenizer, attr, None)
        if value is not None:
            selector_special_ids.add(int(value))

    target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path, trust_remote_code=True)
    if target_tokenizer.pad_token_id is None:
        target_tokenizer.pad_token_id = target_tokenizer.eos_token_id or 0
    target_model_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    mask_token_id = find_mask_token_id(target_tokenizer, target_model_config)
    target_special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
        value = getattr(target_tokenizer, attr, None)
        if value is not None:
            target_special_ids.add(int(value))

    rows = load_manifest(args.eval_data)
    labels_map = {row["uid"]: int(row["label"]) for row in rows}

    prism_backbone = None if str(args.prism_backbone).lower() in {"", "none", "base", "null"} else (
        args.target_backbone if str(args.prism_backbone).lower() == "target" else args.prism_backbone
    )
    prism_model, prism_meta = load_c4_prism(
        prism_checkpoint_path=args.prism_checkpoint,
        base_model_name=args.model_path,
        target_backbone_path=prism_backbone,
        device=device,
        dtype=dtype,
    )
    rare_token_freq_lookup = load_token_frequency_table(args.rare_token_freq_json) if args.rare_token_freq_json else None

    prism_records = collect_clean_prism_signals(
        prism_model,
        rows,
        selector_tokenizer,
        selector_special_ids,
        args.max_length,
        device,
        token_freq_lookup=rare_token_freq_lookup,
    )
    del prism_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_records = collect_target_records(rows=rows, tokenizer=target_tokenizer, special_ids=target_special_ids, max_length=args.max_length)

    target_model = load_raw_backbone(target_model_path, args.target_backbone, device, dtype)
    target_maps = compute_multimask_maps(
        model=target_model,
        prism_records=prism_records,
        target_records=target_records,
        selected_k=args.selected_k,
        selection_mode=args.selection_mode,
        group_size=args.group_size,
        prefix_counts=prefix_counts,
        pad_token_id=int(target_tokenizer.pad_token_id),
        mask_token_id=int(mask_token_id),
        batch_size=args.batch_size,
        device=device,
        selected_indices_fn=selected_indices,
    )
    del target_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    reference_model = load_reference_backbone(target_model_path, args.reference_backbone, device, dtype)
    reference_maps = compute_multimask_maps(
        model=reference_model,
        prism_records=prism_records,
        target_records=target_records,
        selected_k=args.selected_k,
        selection_mode=args.selection_mode,
        group_size=args.group_size,
        prefix_counts=prefix_counts,
        pad_token_id=int(target_tokenizer.pad_token_id),
        mask_token_id=int(mask_token_id),
        batch_size=args.batch_size,
        device=device,
        selected_indices_fn=selected_indices,
    )

    hier_target_maps: dict[str, dict[str, np.ndarray]] = {}
    hier_reference_maps: dict[str, dict[str, np.ndarray]] = {}
    if int(args.group_size) > 0 and not prefix_counts and hier_top_groups and hier_refine_modes:
        stage1_rankings = stage1_group_rankings(target_maps, reference_maps)
        hier_records = build_hierarchical_refine_records(
            prism_records=prism_records,
            stage1_rankings=stage1_rankings,
            top_group_counts=hier_top_groups,
            refine_modes=hier_refine_modes,
        )
        transferred_hier_records: list[tuple[str, str, dict, list[int], np.ndarray]] = []
        for uid, group_tag, record, _selected_pos, idx in hier_records:
            target_record = target_records.get(uid)
            if target_record is None:
                continue
            target_pos, target_idx = transfer_selector_indices_to_target(
                selector_idx=np.asarray(idx, dtype=np.int64),
                selector_valid_count=len(record["valid_pos"]),
                target_valid_pos=target_record["valid_pos"],
            )
            if not target_pos:
                continue
            transferred_hier_records.append((uid, group_tag, target_record, target_pos, target_idx))

        target_model = load_raw_backbone(target_model_path, args.target_backbone, device, dtype)
        hier_target_maps = compute_maps_for_ordered_records(
            model=target_model,
            ordered_records=transferred_hier_records,
            pad_token_id=int(target_tokenizer.pad_token_id),
            mask_token_id=int(mask_token_id),
            batch_size=args.batch_size,
            device=device,
        )
        del target_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        hier_reference_model = load_reference_backbone(target_model_path, args.reference_backbone, device, dtype)
        hier_reference_maps = compute_maps_for_ordered_records(
            model=hier_reference_model,
            ordered_records=transferred_hier_records,
            pad_token_id=int(target_tokenizer.pad_token_id),
            mask_token_id=int(mask_token_id),
            batch_size=args.batch_size,
            device=device,
        )
        del hier_reference_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    del reference_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    features, adaptive_debug, sanity, selector_diagnostics = build_features(
        prism_records=prism_records,
        target_maps=target_maps,
        reference_maps=reference_maps,
        selected_k=args.selected_k,
        selection_mode=args.selection_mode,
        group_size=args.group_size,
        prefix_counts=prefix_counts,
        fixed_huber_clip_values=fixed_huber_clip_values,
        fixed_huber_lower_bound=args.fixed_huber_lower_bound,
        adaptive_huber_clip_values=adaptive_huber_clip_values,
        adaptive_huber_eps=args.adaptive_huber_eps,
        tstat_eps=args.tstat_eps,
        debug_dump_limit=args.debug_dump_limit,
        hier_top_groups=hier_top_groups,
        hier_refine_modes=hier_refine_modes,
        hier_target_maps=hier_target_maps,
        hier_reference_maps=hier_reference_maps,
        save_selector_diagnostics=bool(args.save_selector_diagnostics),
    )
    common_ids = sorted(set(features) & set(labels_map))
    labels = np.asarray([labels_map[uid] for uid in common_ids], dtype=int)
    feature_names = sorted(features[common_ids[0]].keys()) if common_ids else []

    metrics: dict[str, dict] = {}
    scores_out: dict[str, list] = {}
    for feature in feature_names:
        if feature.endswith("_k_actual"):
            continue
        if only_features and feature not in only_features:
            continue
        scores = np.asarray([features[uid][feature] for uid in common_ids], dtype=np.float64)
        result = report_metrics_with_bootstrap(labels, scores, n_bootstraps=args.n_bootstrap_samples, seed=args.bootstrap_seed)
        result["tpr_at_fpr_05"] = tpr_at_fpr(labels, scores, 0.05)
        result["tpr_at_fpr_001"] = tpr_at_fpr(labels, scores, 0.001)
        metrics[feature] = result
        scores_out[feature] = scores.tolist()

    ranked = sorted(metrics.items(), key=lambda item: item[1].get("roc_auc", 0.0), reverse=True)
    best_feature = ranked[0][0] if ranked else None
    best_auc = ranked[0][1].get("roc_auc", 0.0) if ranked else 0.0

    per_sample = []
    for uid in common_ids:
        row = {"uid": uid, "label": labels_map[uid]}
        row.update(features[uid])
        per_sample.append(row)

    summary = {
        "label": "PRISM-selected multi-mask probability MIA",
        "prism_checkpoint": args.prism_checkpoint,
        "selector_model_path": args.model_path,
        "selector_tokenizer_path": tokenizer_source,
        "target_model_path": target_model_path,
        "target_tokenizer_path": target_tokenizer_path,
        "target_backbone": args.target_backbone,
        "prism_backbone": args.prism_backbone,
        "reference_backbone": args.reference_backbone or "base",
        "eval_data": args.eval_data,
        "prism_epoch": prism_meta.get("epoch", "?"),
        "prism_variant": prism_meta.get("variant", "?"),
        "selected_k": int(args.selected_k),
        "selection_mode": args.selection_mode,
        "rare_token_freq_json": args.rare_token_freq_json or None,
        "group_size": int(args.group_size),
        "prefix_counts": prefix_counts or None,
        "fixed_huber_clip_values": fixed_huber_clip_values,
        "fixed_huber_lower_bound": args.fixed_huber_lower_bound,
        "adaptive_huber_clip_values": adaptive_huber_clip_values,
        "adaptive_huber_eps": float(args.adaptive_huber_eps),
        "tstat_eps": float(args.tstat_eps),
        "debug_dump_limit": int(args.debug_dump_limit),
        "hier_top_groups": hier_top_groups or None,
        "hier_refine_modes": hier_refine_modes or None,
        "n_bootstrap_samples": int(args.n_bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "n_sequences": int(len(common_ids)),
        "n_member": int(labels.sum()),
        "n_nonmember": int((1 - labels).sum()),
        "results": metrics,
        "best_feature": best_feature,
        "best_auc": best_auc,
        "sanity_check": {**sanity, "uses_full_p32_mask_for_adaptive_huber": bool(int(sanity["p32_sequence_count"]) == int(sanity["p32_exact_32_count"]))},
        "elapsed_seconds": time.perf_counter() - started_at,
    }
    if prefix_counts:
        curve_family = "target_ref_multimask_true_prob_gap_mean"
        curve_metrics: dict[str, dict[str, float]] = {}
        for prefix_count in prefix_counts:
            feature = f"K{int(args.selected_k)}_{args.selection_mode}_p{int(prefix_count):02d}_{curve_family}"
            if feature in metrics:
                curve_metrics[str(int(prefix_count))] = metrics[feature]
        summary["score_curve_family"] = curve_family
        summary["score_curve_metrics"] = curve_metrics
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(per_sample, indent=2) + "\n", encoding="utf-8")
    if args.save_selector_diagnostics:
        attack_key = f"{args.selection_mode}_p{int(args.selected_k)}"
        token_details = []
        for uid in common_ids:
            detail = selector_diagnostics.get(uid)
            if detail is None:
                continue
            token_details.append({"uid": uid, "label": int(labels_map[uid]), "attack_name": f"{args.selection_mode}_p{int(args.selected_k)}_logprob_gap_hmean", attack_key: detail})
        (out_dir / "per_sample_token_details.json").write_text(json.dumps(token_details, indent=2) + "\n", encoding="utf-8")
    if int(args.debug_dump_limit) > 0:
        (out_dir / "per_sample_adaptive_huber_debug.json").write_text(json.dumps(adaptive_debug, indent=2) + "\n", encoding="utf-8")
    (out_dir / "scores.json").write_text(json.dumps(scores_out, indent=2) + "\n", encoding="utf-8")

    if only_features:
        with (out_dir / "per_sample_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["uid", "label"] + sorted(only_features)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in per_sample:
                writer.writerow({field: row.get(field) for field in fields})
        with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            preferred_metric_fields = [
                "roc_auc", "roc_auc_std", "roc_auc_display", "pr_auc", "pr_auc_std", "pr_auc_display",
                "tpr_at_fpr_10", "tpr_at_fpr_10_std", "tpr_at_fpr_10_display", "tpr_at_fpr_05",
                "tpr_at_fpr_1", "tpr_at_fpr_1_std", "tpr_at_fpr_1_display", "tpr_at_fpr_0_1",
                "tpr_at_fpr_001", "tpr_at_fpr_0_1_std", "tpr_at_fpr_0_1_display", "n_bootstrap_samples",
            ]
            metric_keys = {key for feature in sorted(only_features) if feature in metrics for key in metrics[feature]}
            ordered_metric_fields = [key for key in preferred_metric_fields if key in metric_keys]
            ordered_metric_fields.extend(sorted(metric_keys - set(ordered_metric_fields)))
            fields = ["feature", *ordered_metric_fields]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for feature in sorted(only_features):
                if feature in metrics:
                    writer.writerow({"feature": feature, **metrics[feature]})


if __name__ == "__main__":
    main()
