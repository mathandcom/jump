from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {"text": self.texts[idx], "idx": idx}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_text_records(path: Path, text_field: str) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".jsonl":
        texts = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = extract_text_field(row, text_field)
            if text:
                texts.append(text)
        return texts
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("records", [])
    texts = []
    for row in records:
        text = extract_text_field(row, text_field)
        if text:
            texts.append(text)
    return texts


def extract_text_field(row: Any, text_field: str) -> str | None:
    if isinstance(row, str):
        text = row
    else:
        text = row.get(text_field) or row.get("text") or row.get("x") or row.get("document") or row.get("content")
    text = str(text).strip() if text is not None else ""
    return text or None


def split_train_val(texts: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if len(texts) < 2:
        raise ValueError("Need at least two texts to create train/val splits.")
    indices = np.arange(len(texts))
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    val_count = max(1, int(math.ceil(len(texts) * float(val_fraction))))
    val_indices = set(indices[:val_count].tolist())
    train_texts = [text for idx, text in enumerate(texts) if idx not in val_indices]
    val_texts = [text for idx, text in enumerate(texts) if idx in val_indices]
    return train_texts, val_texts


def load_local_texts(args) -> tuple[list[str], list[str]]:
    train_path = Path(args.train_text_path)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing --train-text-path: {train_path}")
    train_texts = read_text_records(train_path, str(args.text_field))
    if int(args.max_train_samples) > 0:
        train_texts = train_texts[: int(args.max_train_samples)]
    if args.val_text_path:
        val_texts = read_text_records(Path(args.val_text_path), str(args.text_field))
    else:
        train_texts, val_texts = split_train_val(train_texts, float(args.val_fraction), int(args.seed))
    if int(args.max_val_samples) > 0:
        val_texts = val_texts[: int(args.max_val_samples)]
    if not train_texts:
        raise ValueError("No training texts found.")
    if not val_texts:
        raise ValueError("No validation texts found.")
    return train_texts, val_texts


def collate_batch(rows, tokenizer, max_length: int):
    encoded = tokenizer(
        [str(row["text"]) for row in rows],
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        padding=True,
        add_special_tokens=True,
    )
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"].to(dtype=torch.bool)}


def build_special_token_set(tokenizer, mask_token_id: int) -> set[int]:
    return {
        int(token_id)
        for token_id in [
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            int(mask_token_id),
        ]
        if token_id is not None
    }
