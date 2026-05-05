from __future__ import annotations

import json
from pathlib import Path


def load_manifest(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text())
    rows: list[dict] = []
    for g in data["sample_groups"]:
        label = 1 if g["membership"] == "member" else 0
        for s in g["samples"]:
            uid = f"{g['membership']}_{s['sample_idx']}"
            rows.append(
                {
                    "label": label,
                    "membership": g["membership"],
                    "sample_idx": s["sample_idx"],
                    "uid": uid,
                    "text": s["text"],
                }
            )
    return rows
