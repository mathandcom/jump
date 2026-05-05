from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from jump.core.prism_utils import find_mask_token_id
from jump.core.shared.io import load_manifest
from jump.core.shared.metrics import _choose_dtype, _parse_bool, _parse_csv_list, report_metrics_with_bootstrap
from jump.core.shared.models import load_c4_prism, load_raw_backbone
from jump.core.selection.features import compute_features
from jump.core.selection.signals import collect_clean_prism_signals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean PRISM selection + target one-hole entropy MIA")
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--prism_checkpoint", required=True)
    p.add_argument("--target_backbone", required=True)
    p.add_argument("--prism_backbone", default="target")
    p.add_argument("--eval_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--k_values", default="8,16,32,64,128,256,512")
    p.add_argument("--selection_modes", default="quality_bot,entropy_top")
    p.add_argument("--freeze_backbone", default="True")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--hole_batch_size", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_bootstrap_samples", type=int, default=10)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--only_features", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ = _parse_bool(args.freeze_backbone)
    k_values = [int(k) for k in _parse_csv_list(args.k_values)]
    selection_modes = _parse_csv_list(args.selection_modes)
    only_features = set(_parse_csv_list(args.only_features)) if args.only_features else set()
    valid_modes = {"random", "quality_bot", "quality_mean_mid", "quality_median_mid", "rare_token", "entropy_top", "entropy_bot"}
    unknown = [mode for mode in selection_modes if mode not in valid_modes]
    if unknown:
        raise ValueError(f"Unknown selection modes: {unknown}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = _choose_dtype(args, device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    mask_token_id = find_mask_token_id(tokenizer, model_config)
    special_ids: set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id", "mask_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            special_ids.add(int(value))

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
    prism_records = collect_clean_prism_signals(
        prism_model, rows, tokenizer, special_ids, args.max_length, device
    )
    del prism_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_model = load_raw_backbone(args.model_path, args.target_backbone, device, dtype)
    features = compute_features(
        prism_records=prism_records,
        target_model=target_model,
        k_values=k_values,
        selection_modes=selection_modes,
        mask_token_id=mask_token_id,
        device=device,
        hole_batch_size=args.hole_batch_size,
    )
    del target_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    common_ids = sorted(set(features) & set(labels_map))
    labels = np.asarray([labels_map[uid] for uid in common_ids], dtype=int)
    feature_names = sorted(features[common_ids[0]].keys()) if common_ids else []
    metrics: dict[str, dict] = {}
    per_sample = []

    for uid in common_ids:
        row = {"uid": uid, "label": labels_map[uid]}
        row.update(features[uid])
        per_sample.append(row)

    for feature in feature_names:
        if feature.endswith("_k_actual"):
            continue
        if only_features and feature not in only_features:
            continue
        scores = np.asarray([features[uid][feature] for uid in common_ids], dtype=np.float64)
        metrics[feature] = report_metrics_with_bootstrap(
            labels,
            scores,
            n_bootstraps=args.n_bootstrap_samples,
            seed=args.bootstrap_seed,
        )

    summary = {
        "label": "Clean PRISM selection + target one-hole entropy/logit MIA",
        "prism_checkpoint": args.prism_checkpoint,
        "target_backbone": args.target_backbone,
        "prism_backbone": args.prism_backbone,
        "eval_data": args.eval_data,
        "prism_epoch": prism_meta.get("epoch", "?"),
        "prism_variant": prism_meta.get("variant", "?"),
        "results": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "per_sample.json").write_text(json.dumps(per_sample, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
