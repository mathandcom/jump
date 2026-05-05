from __future__ import annotations

import numpy as np
import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    scores = logits.float()
    log_z = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    expected_logit = torch.sum(probs * scores, dim=-1)
    return log_z - expected_logit


@torch.inference_mode()
def collect_clean_prism_signals(
    prism_model: torch.nn.Module,
    rows: list[dict],
    tokenizer,
    special_ids: set[int],
    max_length: int,
    device: torch.device,
    token_freq_lookup: dict[int, int] | None = None,
) -> dict[str, dict]:
    prism_model.eval()
    records: dict[str, dict] = {}
    cached_rows: list[dict] = []
    token_freq: dict[int, int] = {} if token_freq_lookup is None else {
        int(tok): int(count) for tok, count in token_freq_lookup.items()
    }

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
            i
            for i, (tok, attn) in enumerate(zip(clean_ids.tolist(), attn_mask.tolist()))
            if bool(attn) and int(tok) not in special_ids
        ]
        if len(valid_pos) < 2:
            continue
        valid_token_ids = clean_ids[torch.tensor(valid_pos, dtype=torch.long)].tolist()
        if token_freq_lookup is None:
            for tok in valid_token_ids:
                token_freq[int(tok)] = int(token_freq.get(int(tok), 0) + 1)
        cached_rows.append(
            {
                "uid": row["uid"],
                "clean_ids": clean_ids,
                "attn_mask": attn_mask,
                "valid_pos": valid_pos,
                "valid_token_ids": valid_token_ids,
            }
        )

    for item in cached_rows:
        clean_ids = item["clean_ids"]
        attn_mask = item["attn_mask"]
        valid_pos = item["valid_pos"]
        out = prism_model(
            input_ids=clean_ids.unsqueeze(0).to(device),
            attention_mask=attn_mask.unsqueeze(0).to(device),
            compute_unmasking_logits=True,
            compute_quality_scores=True,
        )
        quality_all = out["quality_probs"][0].float().cpu()
        logits_all = out["logits"][0].float().cpu()
        valid_t = torch.tensor(valid_pos, dtype=torch.long)
        valid_logits = logits_all[valid_t]
        valid_ids = clean_ids[valid_t]
        valid_true_logit = valid_logits[torch.arange(len(valid_pos)), valid_ids]
        valid_true_logprob = valid_true_logit - torch.logsumexp(valid_logits, dim=-1)
        records[item["uid"]] = {
            "clean_ids": clean_ids,
            "attn_mask": attn_mask,
            "valid_pos": valid_pos,
            "quality": quality_all[valid_t].numpy(),
            "prism_entropy": entropy_from_logits(valid_logits).numpy(),
            "prism_true_logit": valid_true_logit.numpy(),
            "prism_true_logprob": valid_true_logprob.numpy(),
            "token_frequency": np.asarray(
                [int(token_freq.get(int(tok), 0)) for tok in item["valid_token_ids"]],
                dtype=np.int64,
            ),
        }
    return records


@torch.inference_mode()
def target_onehole_signals_for_positions(
    target_model: torch.nn.Module,
    clean_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    selected_pos: list[int],
    mask_token_id: int,
    device: torch.device,
    hole_batch_size: int,
) -> dict[str, np.ndarray]:
    entropies = []
    true_logits = []
    true_logprobs = []
    clean_ids = clean_ids.to(device)
    attn_mask = attn_mask.to(device=device, dtype=torch.bool)

    for start in range(0, len(selected_pos), int(hole_batch_size)):
        chunk_pos = selected_pos[start : start + int(hole_batch_size)]
        if not chunk_pos:
            continue
        batch_size = len(chunk_pos)
        batch_ids = clean_ids.unsqueeze(0).expand(batch_size, -1).clone()
        batch_mask = attn_mask.unsqueeze(0).expand(batch_size, -1).clone()
        pos_t = torch.tensor(chunk_pos, device=device, dtype=torch.long)
        batch_ids[torch.arange(batch_size, device=device), pos_t] = int(mask_token_id)
        outs = target_model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = outs.logits if hasattr(outs, "logits") else outs[0]
        selected_logits = logits[torch.arange(batch_size, device=device), pos_t, :].float()
        true_ids = clean_ids[pos_t]
        true_logit = selected_logits[torch.arange(batch_size, device=device), true_ids]
        true_logprob = true_logit - torch.logsumexp(selected_logits, dim=-1)
        entropies.append(entropy_from_logits(selected_logits).cpu().numpy())
        true_logits.append(true_logit.cpu().numpy())
        true_logprobs.append(true_logprob.cpu().numpy())

    return {
        "entropy": np.concatenate(entropies) if entropies else np.asarray([], dtype=np.float64),
        "true_logit": np.concatenate(true_logits) if true_logits else np.asarray([], dtype=np.float64),
        "true_logprob": np.concatenate(true_logprobs) if true_logprobs else np.asarray([], dtype=np.float64),
    }
