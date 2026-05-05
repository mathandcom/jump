from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def sample_supervision_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    max_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.full((input_ids.shape[0], int(max_positions)), fill_value=0, dtype=torch.long, device=input_ids.device)
    mask = torch.zeros((input_ids.shape[0], int(max_positions)), dtype=torch.bool, device=input_ids.device)
    for row_idx in range(int(input_ids.shape[0])):
        valid_positions = []
        for pos, token_id in enumerate(input_ids[row_idx].tolist()):
            if not bool(attention_mask[row_idx, pos]) or int(token_id) in special_ids:
                continue
            valid_positions.append(int(pos))
        if not valid_positions:
            continue
        keep = min(len(valid_positions), int(max_positions))
        chosen = torch.randperm(len(valid_positions), device=input_ids.device)[:keep]
        chosen_positions = torch.tensor(valid_positions, device=input_ids.device, dtype=torch.long)[chosen]
        positions[row_idx, :keep] = chosen_positions
        mask[row_idx, :keep] = True
    return positions, mask


def sample_replacement_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    vocab_size: int,
) -> torch.Tensor:
    candidates = []
    for row_idx in range(int(input_ids.shape[0])):
        for pos, token_id in enumerate(input_ids[row_idx].tolist()):
            if not bool(attention_mask[row_idx, pos]) or int(token_id) in special_ids:
                continue
            candidates.append(int(token_id))
    if not candidates:
        candidates = [token_id for token_id in range(vocab_size) if token_id not in special_ids][:1024]
    return torch.tensor(candidates, dtype=torch.long, device=input_ids.device)


def build_corrupted_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    quality_positions: torch.Tensor,
    supervision_mask: torch.Tensor,
    special_ids: set[int],
    vocab_size: int,
) -> torch.Tensor:
    corrupted = input_ids.clone()
    replacement_pool = sample_replacement_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_ids=special_ids,
        vocab_size=int(vocab_size),
    )
    for row_idx in range(int(input_ids.shape[0])):
        valid_cols = torch.nonzero(supervision_mask[row_idx], as_tuple=False).flatten()
        for col in valid_cols.tolist():
            pos = int(quality_positions[row_idx, col].item())
            true_token = int(corrupted[row_idx, pos].item())
            sampled = int(replacement_pool[torch.randint(len(replacement_pool), (1,), device=input_ids.device)].item())
            if sampled == true_token:
                for _ in range(8):
                    sampled = int(replacement_pool[torch.randint(len(replacement_pool), (1,), device=input_ids.device)].item())
                    if sampled != true_token:
                        break
            if sampled == true_token:
                sampled = int((true_token + 1) % int(vocab_size))
                while sampled in special_ids or sampled == true_token:
                    sampled = int((sampled + 1) % int(vocab_size))
            corrupted[row_idx, pos] = sampled
    return corrupted


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not bool(mask.any()):
        return torch.zeros((), device=device, dtype=torch.float32)
    return F.binary_cross_entropy_with_logits(logits[mask].float(), targets[mask].float())


@torch.inference_mode()
def exact_onehole_scores(model, clean_ids, attention_mask, valid_positions, mask_token_id: int, batch_size: int, device):
    scores = []
    clean_ids = clean_ids.to(device)
    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    for start in range(0, len(valid_positions), int(batch_size)):
        chunk_positions = valid_positions[start : start + int(batch_size)]
        if not chunk_positions:
            continue
        inputs = clean_ids.unsqueeze(0).repeat(len(chunk_positions), 1)
        row_positions = torch.tensor(chunk_positions, device=device, dtype=torch.long)
        inputs[torch.arange(len(chunk_positions), device=device), row_positions] = int(mask_token_id)
        attn = attention_mask.unsqueeze(0).repeat(len(chunk_positions), 1)
        outputs = model(
            input_ids=inputs,
            attention_mask=attn,
            quality_positions=None,
            compute_unmasking_logits=True,
            compute_quality_scores=False,
        )
        logits = outputs["logits"]
        logp = F.log_softmax(logits, dim=-1)
        true_ids = clean_ids[row_positions]
        values = logp[torch.arange(len(chunk_positions), device=device), row_positions, true_ids]
        scores.append(values.detach().cpu())
    return torch.cat(scores, dim=0) if scores else torch.empty(0, dtype=torch.float32)


def valid_positions_for_row(clean_ids, attention_mask, special_ids):
    positions = []
    for idx, token_id in enumerate(clean_ids.tolist()):
        if not bool(attention_mask[idx]) or int(token_id) in special_ids:
            continue
        positions.append(int(idx))
    return positions


def simple_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = np.empty_like(x, dtype=np.float64)
    ry = np.empty_like(y, dtype=np.float64)
    rx[np.argsort(x, kind="stable")] = np.arange(x.size, dtype=np.float64)
    ry[np.argsort(y, kind="stable")] = np.arange(y.size, dtype=np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])
