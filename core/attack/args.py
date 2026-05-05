from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PRISM-selected multi-mask probability MIA")
    p.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Base")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--target_model_path", default=None)
    p.add_argument("--target_tokenizer_path", default=None)
    p.add_argument("--prism_checkpoint", required=True)
    p.add_argument("--target_backbone", required=True)
    p.add_argument("--prism_backbone", default="none")
    p.add_argument("--reference_backbone", default="none")
    p.add_argument("--eval_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--selected_k", type=int, default=32)
    p.add_argument("--selection_mode", default="quality_bot")
    p.add_argument("--group_size", type=int, default=0)
    p.add_argument("--prefix_counts", default="")
    p.add_argument("--hier_top_groups", default="")
    p.add_argument("--hier_refine_modes", default="")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n_bootstrap_samples", type=int, default=10)
    p.add_argument("--bootstrap_seed", type=int, default=42)
    p.add_argument("--adaptive_huber_clip_values", default="0.5,1.0,1.5,2.0")
    p.add_argument("--fixed_huber_clip_values", default="0.5")
    p.add_argument("--fixed_huber_lower_bound", type=float, default=None)
    p.add_argument("--adaptive_huber_eps", type=float, default=1e-8)
    p.add_argument("--tstat_eps", type=float, default=1e-8)
    p.add_argument("--debug_dump_limit", type=int, default=0)
    p.add_argument("--only_features", default="")
    p.add_argument("--save_selector_diagnostics", action="store_true")
    p.add_argument("--rare_token_freq_json", default="")
    return p.parse_args()


def load_token_frequency_table(path: str | Path) -> dict[int, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    counts = payload["counts"] if isinstance(payload, dict) and "counts" in payload else payload
    if not isinstance(counts, dict):
        raise ValueError(f"Expected token frequency JSON object at {path}")
    return {int(tok): int(count) for tok, count in counts.items()}
