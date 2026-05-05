from __future__ import annotations

import numpy as np
import torch

from jump.core.prism_utils import score_prism_on_clean_sequence
from jump.core.training.corruption import (
    build_corrupted_batch,
    exact_onehole_scores,
    masked_bce_loss,
    sample_supervision_positions,
    simple_spearman,
    valid_positions_for_row,
)


@torch.inference_mode()
def evaluate_epoch(model, loader, special_ids, mask_token_id: int, vocab_size: int, device, args):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    correlation_values = []
    dev_remaining = int(args.exact_dev_examples)
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.bool)
        quality_positions, supervision_mask = sample_supervision_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            special_ids=special_ids,
            max_positions=int(args.num_supervised_positions),
        )
        if not bool(supervision_mask.any()):
            continue
        corrupt_ids = build_corrupted_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            quality_positions=quality_positions,
            supervision_mask=supervision_mask,
            special_ids=special_ids,
            vocab_size=int(vocab_size),
        )
        clean_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            quality_positions=quality_positions,
            compute_unmasking_logits=False,
            compute_quality_scores=True,
        )["quality_logits"]
        corrupt_logits = model(
            input_ids=corrupt_ids,
            attention_mask=attention_mask,
            quality_positions=quality_positions,
            compute_unmasking_logits=False,
            compute_quality_scores=True,
        )["quality_logits"]
        pos_targets = torch.ones_like(clean_logits, dtype=torch.float32, device=device)
        neg_targets = torch.zeros_like(corrupt_logits, dtype=torch.float32, device=device)
        pos_loss = masked_bce_loss(clean_logits, pos_targets, supervision_mask, device=device)
        neg_loss = masked_bce_loss(corrupt_logits, neg_targets, supervision_mask, device=device)
        batch_loss = 0.5 * (pos_loss + neg_loss)
        total_loss += float(batch_loss.item())

        clean_probs = torch.sigmoid(clean_logits[supervision_mask])
        corrupt_probs = torch.sigmoid(corrupt_logits[supervision_mask])
        total_correct += int((clean_probs >= 0.5).sum().item())
        total_correct += int((corrupt_probs < 0.5).sum().item())
        total_examples += int(clean_probs.numel() + corrupt_probs.numel())

        if dev_remaining > 0:
            for row_idx in range(int(input_ids.shape[0])):
                if dev_remaining <= 0:
                    break
                row_ids = input_ids[row_idx].detach().cpu()
                row_mask = attention_mask[row_idx].detach().cpu()
                valid_positions = valid_positions_for_row(row_ids, row_mask, special_ids)
                if len(valid_positions) < 2:
                    continue
                prism_scores = score_prism_on_clean_sequence(
                    model=model,
                    clean_ids=row_ids,
                    attention_mask=row_mask,
                    valid_positions=valid_positions,
                    device=device,
                )["logprob"].numpy().astype(np.float64)
                exact_scores = exact_onehole_scores(
                    model=model,
                    clean_ids=row_ids,
                    attention_mask=row_mask,
                    valid_positions=valid_positions,
                    mask_token_id=int(mask_token_id),
                    batch_size=int(args.exact_dev_batch_size),
                    device=device,
                ).numpy().astype(np.float64)
                correlation_values.append(simple_spearman(prism_scores, exact_scores))
                dev_remaining -= 1
    return {
        "loss": total_loss / max(1, len(loader)),
        "accuracy": float(total_correct / max(1, total_examples)),
        "supervised_examples": int(total_examples),
        "exact_onehole_spearman_mean": float(np.nanmean(np.asarray(correlation_values, dtype=np.float64)))
        if correlation_values
        else None,
    }
