from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel

from jump.core.attack.score_utils import entropy_from_logits
from jump.core.shared.models import load_raw_backbone


def load_reference_backbone(
    model_path: str,
    reference_backbone: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if reference_backbone is None or str(reference_backbone).lower() in {"", "none", "base", "null"}:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        return model.to(device).eval()
    return load_raw_backbone(model_path, reference_backbone, device, dtype)


def prepare_attention_mask_for_model(
    model: torch.nn.Module,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return attention_mask
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type != "dream":
        return attention_mask

    mask = attention_mask.to(dtype=torch.bool)
    dtype = next(model.parameters()).dtype
    additive = torch.zeros((mask.shape[0], 1, 1, mask.shape[1]), dtype=dtype, device=mask.device)
    additive.masked_fill_(~mask[:, None, None, :], torch.finfo(dtype).min)
    return additive


def collect_target_records(
    rows: list[dict],
    tokenizer,
    special_ids: set[int],
    max_length: int,
) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for row in rows:
        enc = tokenizer(
            row["text"],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        clean_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0].bool()
        valid_pos = [
            i for i, (tok, attn) in enumerate(zip(clean_ids.tolist(), attn_mask.tolist()))
            if bool(attn) and int(tok) not in special_ids
        ]
        if len(valid_pos) < 2:
            continue
        records[row["uid"]] = {"clean_ids": clean_ids, "attn_mask": attn_mask, "valid_pos": valid_pos}
    return records


def transfer_selector_indices_to_target(
    selector_idx: np.ndarray,
    selector_valid_count: int,
    target_valid_pos: list[int],
) -> tuple[list[int], np.ndarray]:
    if selector_valid_count <= 0 or not target_valid_pos:
        return [], np.asarray([], dtype=np.int64)

    target_valid_count = len(target_valid_pos)
    mapped_positions: list[int] = []
    mapped_selector_idx: list[int] = []
    used_target_ord: set[int] = set()

    for raw_idx in np.asarray(selector_idx, dtype=np.int64).tolist():
        if selector_valid_count == 1:
            target_ord = 0
        else:
            ratio = float(raw_idx) / float(max(1, selector_valid_count - 1))
            target_ord = int(round(ratio * float(max(0, target_valid_count - 1))))
        target_ord = min(max(target_ord, 0), target_valid_count - 1)

        if target_ord in used_target_ord:
            for radius in range(1, target_valid_count):
                left = target_ord - radius
                right = target_ord + radius
                if left >= 0 and left not in used_target_ord:
                    target_ord = left
                    break
                if right < target_valid_count and right not in used_target_ord:
                    target_ord = right
                    break
            else:
                continue

        used_target_ord.add(target_ord)
        mapped_positions.append(int(target_valid_pos[target_ord]))
        mapped_selector_idx.append(int(raw_idx))

    return mapped_positions, np.asarray(mapped_selector_idx, dtype=np.int64)


def batch_multimask_signals(
    model: torch.nn.Module,
    batch_records: list[tuple[str, str, dict, list[int], np.ndarray]],
    pad_token_id: int,
    mask_token_id: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    if not batch_records:
        return {}

    max_len = max(int(record["clean_ids"].shape[0]) for _, _, record, _, _ in batch_records)
    batch_size = len(batch_records)
    input_ids = torch.full((batch_size, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for row_idx, (_, _, record, _, _) in enumerate(batch_records):
        clean_ids = record["clean_ids"]
        attn_mask = record["attn_mask"]
        seq_len = int(clean_ids.shape[0])
        input_ids[row_idx, :seq_len] = clean_ids
        attention_mask[row_idx, :seq_len] = attn_mask

    masked_ids = input_ids.clone()
    for row_idx, (_, _, _, selected_pos, _) in enumerate(batch_records):
        if selected_pos:
            masked_ids[row_idx, torch.tensor(selected_pos, dtype=torch.long)] = int(mask_token_id)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    masked_ids = masked_ids.to(device)
    model_attention_mask = prepare_attention_mask_for_model(model, attention_mask)

    with torch.no_grad():
        out = model(input_ids=masked_ids, attention_mask=model_attention_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]

    maps: dict[str, dict[str, np.ndarray]] = {}
    for row_idx, (uid, group_tag, _, selected_pos, idx) in enumerate(batch_records):
        pos_t = torch.tensor(selected_pos, device=device, dtype=torch.long)
        true_t = input_ids[row_idx, pos_t]
        selected_logits = logits[row_idx, pos_t, :].float()
        true_logits_t = selected_logits[torch.arange(len(selected_pos), device=device), true_t]
        true_logprob_t = true_logits_t - torch.logsumexp(selected_logits, dim=-1)
        maps[f"{uid}::{group_tag}"] = {
            "uid": uid,
            "group_tag": group_tag,
            "idx": np.asarray(idx, dtype=np.int64),
            "selected_positions": np.asarray(selected_pos, dtype=np.int64),
            "selected_token_ids": true_t.cpu().numpy(),
            "entropy": entropy_from_logits(selected_logits).cpu().numpy(),
            "true_logit": true_logits_t.cpu().numpy(),
            "true_logprob": true_logprob_t.cpu().numpy(),
            "true_prob": torch.exp(true_logprob_t).cpu().numpy(),
        }
    return maps


@torch.inference_mode()
def compute_maps_for_ordered_records(
    model: torch.nn.Module,
    ordered_records: list[tuple[str, str, dict, list[int], np.ndarray]],
    pad_token_id: int,
    mask_token_id: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    maps: dict[str, dict[str, np.ndarray]] = {}
    for start in range(0, len(ordered_records), int(batch_size)):
        batch_records = ordered_records[start:start + int(batch_size)]
        maps.update(
            batch_multimask_signals(
                model=model,
                batch_records=batch_records,
                pad_token_id=pad_token_id,
                mask_token_id=mask_token_id,
                device=device,
            )
        )
    return maps


@torch.inference_mode()
def compute_multimask_maps(
    model: torch.nn.Module,
    prism_records: dict[str, dict],
    target_records: dict[str, dict],
    selected_k: int,
    selection_mode: str,
    group_size: int,
    prefix_counts: list[int],
    pad_token_id: int,
    mask_token_id: int,
    batch_size: int,
    device: torch.device,
    selected_indices_fn,
) -> dict[str, dict[str, np.ndarray]]:
    ordered_records: list[tuple[str, str, dict, list[int], np.ndarray]] = []
    for uid, record in prism_records.items():
        target_record = target_records.get(uid)
        if target_record is None:
            continue
        selections = selected_indices_fn(record, int(selected_k), [selection_mode])
        if not selections:
            continue
        _, idx = selections[0]
        selected_idx = np.asarray(idx, dtype=np.int64)
        selected_pos, selected_idx = transfer_selector_indices_to_target(
            selector_idx=selected_idx,
            selector_valid_count=len(record["valid_pos"]),
            target_valid_pos=target_record["valid_pos"],
        )
        if not selected_pos:
            continue
        if prefix_counts:
            for prefix_count in prefix_counts:
                prefix_count = min(len(selected_pos), max(1, int(prefix_count)))
                group_tag = f"p{prefix_count:02d}"
                ordered_records.append((uid, group_tag, target_record, selected_pos[:prefix_count], selected_idx[:prefix_count]))
        elif int(group_size) > 0:
            for group_start in range(0, len(selected_pos), int(group_size)):
                group_end = min(len(selected_pos), group_start + int(group_size))
                group_tag = f"g{group_start // int(group_size) + 1:02d}"
                group_pos = selected_pos[group_start:group_end]
                group_idx = selected_idx[group_start:group_end]
                if group_pos:
                    ordered_records.append((uid, group_tag, target_record, group_pos, group_idx))
        else:
            ordered_records.append((uid, "all", target_record, selected_pos, selected_idx))

    return compute_maps_for_ordered_records(
        model=model,
        ordered_records=ordered_records,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        batch_size=batch_size,
        device=device,
    )
