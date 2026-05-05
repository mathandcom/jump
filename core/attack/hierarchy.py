from __future__ import annotations

from itertools import combinations

import numpy as np


def stage1_group_rankings(
    target_maps: dict[str, dict[str, np.ndarray]],
    reference_maps: dict[str, dict[str, np.ndarray]],
) -> dict[str, list[dict]]:
    rankings: dict[str, list[dict]] = {}
    for key in sorted(set(target_maps) & set(reference_maps)):
        target = target_maps[key]
        reference = reference_maps[key]
        group_tag = str(target["group_tag"])
        if not group_tag.startswith("g"):
            continue
        uid = str(target["uid"])
        score = float(
            np.mean(
                np.asarray(target["true_prob"], dtype=np.float64)
                - np.asarray(reference["true_prob"], dtype=np.float64)
            )
        )
        rankings.setdefault(uid, []).append(
            {"group_tag": group_tag, "score": score, "idx": np.asarray(target["idx"], dtype=np.int64)}
        )
    for uid, rows in rankings.items():
        rows.sort(key=lambda row: (row["score"], row["group_tag"]), reverse=True)
    return rankings


def build_hierarchical_refine_records(
    prism_records: dict[str, dict],
    stage1_rankings: dict[str, list[dict]],
    top_group_counts: list[int],
    refine_modes: list[str],
) -> list[tuple[str, str, dict, list[int], np.ndarray]]:
    ordered_records: list[tuple[str, str, dict, list[int], np.ndarray]] = []
    for uid, ranked_groups in stage1_rankings.items():
        record = prism_records[uid]
        valid_pos = record["valid_pos"]
        for top_n in top_group_counts:
            chosen_groups = ranked_groups[:min(int(top_n), len(ranked_groups))]
            if not chosen_groups:
                continue
            for mode in refine_modes:
                if mode == "onehole":
                    for group in chosen_groups:
                        group_idx = np.asarray(group["idx"], dtype=np.int64)
                        group_pos = [int(valid_pos[int(i)]) for i in group_idx]
                        for token_ord, (token_idx, token_pos) in enumerate(zip(group_idx.tolist(), group_pos), start=1):
                            group_tag = f"h|t{int(top_n):02d}|onehole|{group['group_tag']}|u{token_ord:02d}"
                            ordered_records.append(
                                (uid, group_tag, record, [int(token_pos)], np.asarray([int(token_idx)], dtype=np.int64))
                            )
                elif mode == "twohole":
                    for group in chosen_groups:
                        group_idx = np.asarray(group["idx"], dtype=np.int64)
                        if len(group_idx) < 2:
                            continue
                        group_pos = [int(valid_pos[int(i)]) for i in group_idx]
                        for pair_ord, (left, right) in enumerate(combinations(range(len(group_idx)), 2), start=1):
                            group_tag = f"h|t{int(top_n):02d}|twohole|{group['group_tag']}|p{pair_ord:02d}"
                            ordered_records.append(
                                (
                                    uid,
                                    group_tag,
                                    record,
                                    [int(group_pos[left]), int(group_pos[right])],
                                    np.asarray([int(group_idx[left]), int(group_idx[right])], dtype=np.int64),
                                )
                            )
                else:
                    raise ValueError(f"Unsupported hier_refine_mode: {mode}")
    return ordered_records
