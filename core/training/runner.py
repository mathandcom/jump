from __future__ import annotations

import math
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from jump.core.prism_utils import (
    build_prism_model,
    find_mask_token_id,
    save_prism_checkpoint,
    write_json,
)
from jump.core.training.args import parse_args
from jump.core.training.corruption import build_corrupted_batch, masked_bce_loss, sample_supervision_positions
from jump.core.training.data import (
    TextDataset,
    build_special_token_set,
    collate_batch,
    load_local_texts,
    set_seed,
)
from jump.core.training.evaluation import evaluate_epoch


def _metadata(args, tokenizer_source: str, mask_token_id: int, lora_modules: list[str]) -> dict:
    return {
        "base_model": str(args.base_model),
        "tokenizer_source": str(tokenizer_source),
        "backbone_checkpoint": str(args.backbone_checkpoint) if args.backbone_checkpoint else None,
        "train_text_path": str(args.train_text_path),
        "val_text_path": str(args.val_text_path) if args.val_text_path else None,
        "text_field": str(args.text_field),
        "max_length": int(args.max_length),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_patterns": ["q_proj", "k_proj", "v_proj", "Wqkv", "c_attn"],
        "mask_token_id": int(mask_token_id),
        "replaced_lora_modules": lora_modules,
        "training_objective": "replaced_token_quality",
        "lambda_mdm": float(args.lambda_mdm),
        "num_supervised_positions": int(args.num_supervised_positions),
        "precision": str(args.precision),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(args.seed))

    mixed_precision = None if args.precision == "no" else str(args.precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.grad_accum),
        mixed_precision=mixed_precision,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.precision == "bf16":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    elif args.precision == "fp16":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.float32

    tokenizer_source = args.tokenizer_path or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    model, _, lora_modules = build_prism_model(
        base_model_name=str(args.base_model),
        backbone_checkpoint=args.backbone_checkpoint,
        device=device,
        dtype=dtype,
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        freeze_base=True,
    )
    mask_token_id = find_mask_token_id(tokenizer, model.config)
    special_ids = build_special_token_set(tokenizer, mask_token_id)
    vocab_size = int(getattr(model.config, "vocab_size"))

    train_texts, val_texts = load_local_texts(args)
    train_dataset = TextDataset(train_texts)
    val_dataset = TextDataset(val_texts)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=int(args.seed),
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        seed=int(args.seed),
        drop_last=False,
    )
    collate_fn = lambda rows: collate_batch(rows, tokenizer, int(args.max_length))
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), sampler=train_sampler, num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), sampler=val_sampler, num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)

    history = []
    best_metric = -float("inf")
    best_summary = None
    for epoch_idx in range(int(args.epochs)):
        train_sampler.set_epoch(epoch_idx)
        model.train()
        running_loss = 0.0
        running_examples = 0
        running_correct = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
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
            pos_targets = torch.ones_like(quality_positions, dtype=torch.float32, device=device)
            neg_targets = torch.zeros_like(quality_positions, dtype=torch.float32, device=device)

            with accelerator.accumulate(model):
                clean_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    quality_positions=quality_positions,
                    compute_unmasking_logits=False,
                    compute_quality_scores=True,
                )["quality_logits"]
                pos_loss = masked_bce_loss(clean_logits, pos_targets, supervision_mask, device=device)
                accelerator.backward(0.5 * pos_loss)

                corrupt_logits = model(
                    input_ids=corrupt_ids,
                    attention_mask=attention_mask,
                    quality_positions=quality_positions,
                    compute_unmasking_logits=False,
                    compute_quality_scores=True,
                )["quality_logits"]
                neg_loss = masked_bce_loss(corrupt_logits, neg_targets, supervision_mask, device=device)
                accelerator.backward(0.5 * neg_loss)

                total_loss = 0.5 * (pos_loss.detach() + neg_loss.detach())
                if float(args.lambda_mdm) > 0.0:
                    raise ValueError("Optional MDM auxiliary loss is intentionally disabled in the default PRISM trainer.")

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1

            running_loss += float(total_loss.item())
            clean_probs = torch.sigmoid(clean_logits[supervision_mask].detach())
            corrupt_probs = torch.sigmoid(corrupt_logits[supervision_mask].detach())
            running_correct += int((clean_probs >= 0.5).sum().item())
            running_correct += int((corrupt_probs < 0.5).sum().item())
            running_examples += int(clean_probs.numel() + corrupt_probs.numel())

        val_metrics = evaluate_epoch(
            model=accelerator.unwrap_model(model),
            loader=val_loader,
            special_ids=special_ids,
            mask_token_id=int(mask_token_id),
            vocab_size=int(vocab_size),
            device=device,
            args=args,
        )
        epoch_summary = {
            "epoch": int(epoch_idx + 1),
            "train_loss": float(running_loss / max(1, len(train_loader))),
            "train_accuracy": float(running_correct / max(1, running_examples)),
            "train_supervised_examples": int(running_examples),
            "optimizer_steps": int(optimizer_steps),
            "val": val_metrics,
        }
        if is_main:
            history.append(epoch_summary)
            model_metric = float(
                val_metrics["exact_onehole_spearman_mean"]
                if val_metrics["exact_onehole_spearman_mean"] is not None
                and not math.isnan(float(val_metrics["exact_onehole_spearman_mean"]))
                else val_metrics["accuracy"]
            )
            if model_metric > best_metric:
                best_metric = model_metric
                best_summary = epoch_summary
                save_prism_checkpoint(out_dir / "checkpoint_best.pt", model=accelerator.unwrap_model(model), metadata=_metadata(args, tokenizer_source, int(mask_token_id), lora_modules), summary=epoch_summary)
            if args.save_every_epoch:
                save_prism_checkpoint(out_dir / f"checkpoint_epoch_{epoch_idx + 1}.pt", model=accelerator.unwrap_model(model), metadata=_metadata(args, tokenizer_source, int(mask_token_id), lora_modules), summary=epoch_summary)

        accelerator.wait_for_everyone()

    if is_main:
        save_prism_checkpoint(
            out_dir / "checkpoint_last.pt",
            model=accelerator.unwrap_model(model),
            metadata=_metadata(args, tokenizer_source, int(mask_token_id), lora_modules),
            summary=history[-1] if history else None,
        )
        summary = {"args": vars(args), "history": history, "best": best_summary, "train_examples": len(train_texts), "val_examples": len(val_texts)}
        write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
