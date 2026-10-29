"""Deterministic, group-disjoint train/val/test assignment and cheap row sampling."""

from __future__ import annotations

import bisect
import hashlib
import random
import re
from collections import defaultdict


def assign_split(group: str, val_frac: float, test_frac: float, seed: str = "hinglish-v1") -> str:
    """"train", "val" or "test" for every utterance of `group`.

    Hashing the group (speaker or call), not the utterance, keeps a speaker's
    utterances on one side of the split; sha256 rather than hash() keeps it
    stable across processes and runs.
    """
    digest = hashlib.sha256(f"{seed}\0{group}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def group_key(path: str, pattern: str | None, fallback: str) -> str:
    """First capture group of `pattern` in the file name, else `fallback`."""
    if pattern:
        m = re.search(pattern, path.rsplit("/", 1)[-1])
        if m:
            return m.group(1)
    return fallback


def sample_rows_by_row_group(
    rows: list[int], bounds: list[tuple[int, int]], n: int, seed: int = 0
) -> list[int]:
    """Up to `n` of `rows`, drawn from as few parquet row groups as possible.

    Each row group is read whole (~88 MB for Hindi-Transcription-DS), so a
    uniform sample of rows would touch nearly every group. Whole groups in a
    seeded random order keep the sample spread across the file while reading
    a handful of groups.
    """
    starts = [s for s, _ in bounds]
    by_group: dict[int, list[int]] = defaultdict(list)
    for r in sorted(rows):
        by_group[bisect.bisect_right(starts, r) - 1].append(r)
    order = sorted(by_group)
    random.Random(seed).shuffle(order)
    picked: list[int] = []
    for g in order:
        if len(picked) >= n:
            break
        picked.extend(by_group[g])
    return sorted(picked[:n])
