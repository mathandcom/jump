from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel

try:  # pragma: no cover
    from jump.core.prism_utils import (
        LLaDAPrismModel,
        build_prism_model,
        load_prism_checkpoint,
    )
except ModuleNotFoundError:  # pragma: no cover
    from ..prism_utils import LLaDAPrismModel, build_prism_model, load_prism_checkpoint


def load_c4_prism(
    prism_checkpoint_path: str,
    base_model_name: str,
    target_backbone_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LLaDAPrismModel, dict]:
    payload = torch.load(prism_checkpoint_path, map_location="cpu")
    fmt = payload.get("format", "llada_prism_v1")
    metadata = dict(payload.get("metadata", {}))

    if fmt == "llada_prism_v1_full":
        inner, _, _ = build_prism_model(
            base_model_name=base_model_name,
            backbone_checkpoint=None,
            device=device,
            dtype=dtype,
            lora_rank=0,
            freeze_base=False,
        )
        sd = payload["state_dict"]
        sd_cast = {k: (v if "quality_head" in k else v.to(dtype)) for k, v in sd.items()}
        inner.load_state_dict(sd_cast, strict=False)
    else:
        inner, metadata, _ = load_prism_checkpoint(
            prism_checkpoint_path=prism_checkpoint_path,
            base_model_name=base_model_name,
            backbone_checkpoint=target_backbone_path,
            device=device,
            dtype=dtype,
        )

    inner = inner.to(device)
    inner.eval()
    return inner, metadata


def load_raw_backbone(
    model_path: str,
    backbone_checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    checkpoint_path = Path(str(backbone_checkpoint))
    if checkpoint_path.is_dir():
        backbone = AutoModel.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        return backbone.to(device).eval()

    backbone = AutoModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    payload = torch.load(backbone_checkpoint, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Backbone load mismatch. missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return backbone.to(device).eval()
