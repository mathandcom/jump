#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel


def resolve_hidden_size(config: Any) -> int:
    for key in ("hidden_size", "d_model", "n_embd", "model_dim"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise ValueError("Could not determine hidden size from model config.")


def find_mask_token_id(tokenizer, model_config) -> int:
    if tokenizer.mask_token_id is not None:
        return int(tokenizer.mask_token_id)
    mask_token_id = getattr(model_config, "mask_token_id", None)
    if mask_token_id is not None:
        return int(mask_token_id)
    for candidate in ("<mask>", "[MASK]", "<|mask|>"):
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            return int(token_id)
    raise ValueError("Could not determine a mask token id for LLaDA.")


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}")
        if int(rank) <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base_layer.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Outside autocast (e.g. standalone inference), backbone runs in fp16 but
        # LoRA weights are float32.  Cast explicitly to avoid dtype mismatch.
        weight_dtype = self.lora_a.weight.dtype
        if inputs.dtype != weight_dtype:
            lora_out = self.lora_b(self.dropout(self.lora_a(inputs.to(weight_dtype)))) * self.scaling
            return self.base(inputs) + lora_out.to(inputs.dtype)
        return self.base(inputs) + self.lora_b(self.dropout(self.lora_a(inputs))) * self.scaling


def _replace_child_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora_modules(
    model: nn.Module,
    target_patterns: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "Wqkv", "c_attn"),
    rank: int = 0,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> list[str]:
    if int(rank) <= 0:
        return []
    replacements: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(pattern in module_name for pattern in target_patterns):
            continue
        _replace_child_module(
            root=model,
            module_name=module_name,
            new_module=LoRALinear(module, rank=int(rank), alpha=float(alpha), dropout=float(dropout)),
        )
        replacements.append(module_name)
    return replacements


class PrismQualityHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.GELU(),
            nn.Linear(int(hidden_size), 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)


class LLaDAPrismModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.hidden_size = int(hidden_size or resolve_hidden_size(backbone.config))
        self.quality_head = PrismQualityHead(self.hidden_size)

    def forward(
        self,
        *args,
        quality_positions: torch.Tensor | None = None,
        compute_unmasking_logits: bool = False,
        compute_quality_scores: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs["use_cache"] = False
        kwargs["output_hidden_states"] = False

        captured: dict[str, torch.Tensor] = {}
        hook_handle = None
        use_hidden_state_fallback = False
        if compute_quality_scores:
            backbone_model = getattr(self.backbone, "model", None)
            transformer = getattr(backbone_model, "transformer", None)
            ln_f = getattr(transformer, "ln_f", None)
            if ln_f is not None:
                def _capture_final_hidden(_module, _inputs, output):
                    captured["final_hidden"] = output

                hook_handle = ln_f.register_forward_hook(_capture_final_hidden)
            else:
                # Dream-style backbones expose the final representation via
                # output_hidden_states/last_hidden_state instead of a GPT-style ln_f.
                kwargs["output_hidden_states"] = True
                use_hidden_state_fallback = True

        try:
            outputs = self.backbone(*args, **kwargs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        logits = (outputs.logits if hasattr(outputs, "logits") else outputs[0]) if compute_unmasking_logits else None

        quality_logits = None
        quality_probs = None
        if compute_quality_scores:
            final_hidden = captured.get("final_hidden")
            if final_hidden is None and use_hidden_state_fallback:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    final_hidden = hidden_states[-1]
                else:
                    final_hidden = getattr(outputs, "last_hidden_state", None)
            if final_hidden is None:
                raise ValueError("Backbone did not expose the final hidden state needed for the PRISM head.")
            if quality_positions is not None:
                if quality_positions.ndim == 1:
                    batch_indices = torch.arange(final_hidden.shape[0], device=final_hidden.device)
                    final_hidden = final_hidden[batch_indices, quality_positions]
                elif quality_positions.ndim == 2:
                    batch_indices = torch.arange(final_hidden.shape[0], device=final_hidden.device).unsqueeze(1)
                    final_hidden = final_hidden[batch_indices, quality_positions]
                else:
                    raise ValueError("quality_positions must be rank-1 or rank-2.")
            # Cast hidden states to quality_head's dtype (float32 during training,
            # matches backbone fp16 during standalone inference after checkpoint load).
            head_dtype = next(self.quality_head.parameters()).dtype
            quality_logits = self.quality_head(final_hidden.to(dtype=head_dtype))
            quality_probs = torch.sigmoid(quality_logits.float())

        return {
            "logits": logits,
            "quality_logits": quality_logits,
            "quality_probs": quality_probs,
        }


def freeze_non_prism_parameters(model: LLaDAPrismModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("quality_head.") or ".lora_" in name


def prism_trainable_state_dict(model: LLaDAPrismModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    kept: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("quality_head.") or ".lora_a." in key or ".lora_b." in key:
            kept[key] = value.detach().cpu()
    return kept


def save_prism_checkpoint(
    path: Path,
    model: LLaDAPrismModel,
    metadata: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "llada_prism_v1",
        "metadata": metadata,
        "prism_state_dict": prism_trainable_state_dict(model),
    }
    if summary is not None:
        payload["summary"] = summary
    torch.save(payload, path)


def load_backbone_model(
    base_model_name: str,
    checkpoint_path: str | Path | None,
    device: torch.device,
    dtype: torch.dtype,
    low_cpu_mem_usage: bool = True,
) -> tuple[nn.Module, dict[str, Any] | None]:
    """Load LLaDA backbone.

    low_cpu_mem_usage=True (default) uses PyTorch's memory-mapped weight
    loading so each process only buffers one tensor at a time in CPU RAM
    (~few hundred MB) instead of staging the full 16 GB model before moving
    to GPU.  This is critical when running 4 DDP processes under a 32 GB
    system-RAM SLURM --mem limit.
    """
    if checkpoint_path is not None and Path(str(checkpoint_path)).is_dir():
        model = AutoModel.from_pretrained(
            str(checkpoint_path),
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=bool(low_cpu_mem_usage),
        )
        return model.to(device), None

    model = AutoModel.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        low_cpu_mem_usage=bool(low_cpu_mem_usage),
    )
    checkpoint_payload = None
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload["model"] if isinstance(checkpoint_payload, dict) and "model" in checkpoint_payload else checkpoint_payload
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Unexpected checkpoint load mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    return model.to(device), checkpoint_payload


def build_prism_model(
    base_model_name: str,
    backbone_checkpoint: str | Path | None,
    device: torch.device,
    dtype: torch.dtype,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    lora_target_patterns: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "Wqkv", "c_attn"),
    freeze_base: bool = True,
) -> tuple[LLaDAPrismModel, dict[str, Any] | None, list[str]]:
    backbone, checkpoint_payload = load_backbone_model(
        base_model_name=base_model_name,
        checkpoint_path=backbone_checkpoint,
        device=device,
        dtype=dtype,
    )
    replaced = inject_lora_modules(
        model=backbone,
        target_patterns=tuple(lora_target_patterns),
        rank=int(lora_rank),
        alpha=float(lora_alpha),
        dropout=float(lora_dropout),
    )
    model = LLaDAPrismModel(backbone=backbone).to(device)
    if freeze_base:
        freeze_non_prism_parameters(model)
    return model, checkpoint_payload, replaced


def load_prism_checkpoint(
    prism_checkpoint_path: str | Path,
    base_model_name: str | None,
    backbone_checkpoint: str | Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[LLaDAPrismModel, dict[str, Any], dict[str, Any] | None]:
    payload = torch.load(prism_checkpoint_path, map_location="cpu")
    metadata = dict(payload.get("metadata", {}))
    resolved_base_model = str(base_model_name or metadata.get("base_model"))
    if not resolved_base_model:
        raise ValueError("A base model must be provided either directly or inside the PRISM checkpoint metadata.")
    resolved_backbone_checkpoint = backbone_checkpoint if backbone_checkpoint is not None else metadata.get("backbone_checkpoint")
    lora_rank = int(metadata.get("lora_rank", 0))
    lora_alpha = float(metadata.get("lora_alpha", 16.0))
    lora_dropout = float(metadata.get("lora_dropout", 0.0))
    lora_target_patterns = tuple(metadata.get("lora_target_patterns", ("q_proj", "k_proj", "v_proj", "Wqkv", "c_attn")))
    model, checkpoint_payload, _ = build_prism_model(
        base_model_name=resolved_base_model,
        backbone_checkpoint=resolved_backbone_checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_patterns=lora_target_patterns,
        freeze_base=True,
    )
    missing, unexpected = model.load_state_dict(payload["prism_state_dict"], strict=False)
    unexpected = [item for item in unexpected if "base." not in item]
    allowed_missing = [
        item
        for item in missing
        if not (item.startswith("quality_head.") or ".lora_a." in item or ".lora_b." in item)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected PRISM checkpoint mismatch. missing={missing[:10]} unexpected={unexpected[:10]}")
    if any(item.startswith("quality_head.") or ".lora_a." in item or ".lora_b." in item for item in missing):
        raise RuntimeError(
            f"PRISM checkpoint is missing trainable PRISM weights. missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    model.eval()
    return model, metadata, checkpoint_payload


@torch.inference_mode()
def score_prism_on_clean_sequence(
    model: LLaDAPrismModel,
    clean_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    valid_positions: list[int],
    device: torch.device,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    inputs = clean_ids.unsqueeze(0).to(device)
    attn = attention_mask.unsqueeze(0).to(device=device, dtype=torch.bool)
    outputs = model(input_ids=inputs, attention_mask=attn)
    quality_probs = outputs["quality_probs"][0]
    positions = torch.tensor(valid_positions, device=device, dtype=torch.long)
    selected_probs = quality_probs[positions].clamp(min=float(eps), max=1.0 - float(eps))
    return {
        "quality_prob": selected_probs.detach().cpu(),
        "logprob": torch.log(selected_probs).detach().cpu(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
