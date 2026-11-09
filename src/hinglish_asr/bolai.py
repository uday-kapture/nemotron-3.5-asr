"""kapturecx/bolAIndia: per-file stats and which call rows train.

The dataset (gated, ~12.8 TB) is never downloaded whole. `stats/*.json`
describe every parquet file (rows, hours, language and transcript-vendor mix),
so `plan` chooses files without reading audio. Every file holds 16 kHz mono
FLAC chunks, already VAD-cut per call side, which `convert` copies into tar
shards byte for byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# source_collection values that mark someone else's held-out data.
TEST_LIKE_COLLECTIONS = frozenset({"valid", "validation", "dev", "test", "eval", "benchmark"})

# Language code -> the prompt model's language key (it has no en-IN).
TARGET_LANG = {"hi": "hi-IN", "en": "en-US"}

COLUMNS = (
    "audio",
    "text",
    "provider",
    "language_code",
    "is_unintelligible",
    "confidence",
    "duration_s",
    "source_collection",
    "call_direction",
    "to_phone_hash",
    "from_phone_hash",
    "call_key",
    "chunk_id",
    "channel",
    "client_id",
)


def load_stats(stats_dir: str | Path) -> dict[str, dict]:
    """Per-file stats keyed by the file's path at the pinned revision.

    Several hosts' stats list the same file with identical numbers, and files
    first written under a vendor directory were later renamed;
    `config-path-aliases.json` maps old paths to current ones.
    """
    stats_dir = Path(stats_dir)
    aliases_file = stats_dir / "config-path-aliases.json"
    aliases = json.loads(aliases_file.read_text())["path_aliases"] if aliases_file.exists() else {}
    out: dict[str, dict] = {}
    for f in sorted(stats_dir.glob("*.json")):
        for path, shard in json.loads(f.read_text()).get("shards", {}).items():
            out[aliases.get(path, path)] = shard
    return out


def fetch_stats(repo: str, revision: str, dest: str | Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo, repo_type="dataset", revision=revision, allow_patterns=["stats/*.json"], local_dir=dest
    )
    return Path(dest) / "stats"


def subset_of(path: str) -> str:
    """"data/combined/combined-....parquet" -> "combined"."""
    parts = path.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "data" else ""


def customer_key(row: dict) -> str:
    """The customer on the call; splits keep each customer on one side.

    Outbound calls dial the customer (to_phone_hash), inbound calls come from
    them (from_phone_hash). Without either, the call is the group.
    """
    direction = (row.get("call_direction") or "").lower()
    phone = {"outbound": row.get("to_phone_hash"), "inbound": row.get("from_phone_hash")}.get(direction)
    if phone:
        return f"phone:{phone}"
    return f"call:{row.get('call_key') or row.get('chunk_id')}"


@dataclass(frozen=True)
class RowFilter:
    providers: tuple[str, ...] | None = None  # `provider` column values; None = every vendor and human labels
    languages: tuple[str, ...] = ("hi", "en")
    min_duration: float = 0.1
    max_duration: float = 20.0
    # One threshold for every vendor, or {provider: threshold}: vendors' confidences
    # are not on one scale. Rows without a confidence (human labels) are not gated.
    min_confidence: float | dict[str, float] = 0.0
    channels: tuple[str, ...] | None = None
    # Empty clips (no transcript, or is_unintelligible) become empty-target
    # candidates for `verify_empty` instead of being dropped.
    empty_targets: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> RowFilter:
        channels, providers = cfg.get("channels"), cfg.get("providers")
        return cls(
            providers=tuple(providers) if providers else None,
            languages=tuple(cfg["languages"]),
            min_duration=cfg["min_duration"],
            max_duration=cfg["max_duration"],
            min_confidence=cfg["min_confidence"],
            channels=tuple(channels) if channels else None,
            empty_targets=bool(cfg.get("empty_targets")),
        )

    def confidence_floor(self, provider: str | None) -> float:
        if isinstance(self.min_confidence, dict):
            return self.min_confidence.get(provider, 0.0)
        return self.min_confidence

    def _common(self, row: dict) -> str | None:
        if (row.get("source_collection") or "").lower() in TEST_LIKE_COLLECTIONS:
            return "test_collection"
        if self.channels and row.get("channel") not in self.channels:
            return "channel"
        duration = row.get("duration_s") or 0.0
        if duration < self.min_duration:
            return "too_short"
        if duration > self.max_duration:
            return "too_long"
        return None

    def empty_target(self, row: dict) -> bool:
        """A clip the vendor heard nothing in: a candidate empty target.

        Not proof of silence: every chunk is VAD-cut, so VAD speech ratios of empty
        and transcribed clips overlap. `verify_empty` keeps only clips a second
        model also leaves empty. Language is not checked: such rows often have none.
        """
        if not self.empty_targets:
            return False
        if self.providers and row.get("provider") not in self.providers:
            return False
        if not (row.get("is_unintelligible") or not (row.get("text") or "").strip()):
            return False
        return self._common(row) is None

    def reason(self, row: dict) -> str | None:
        """Why `row` is not used, or None to keep it.

        In bolAIndia an empty transcript, is_unintelligible and confidence 0
        come together (a quarter of Deepgram rows in the sampled file).
        """
        if self.providers and row.get("provider") not in self.providers:
            return "other_vendor"
        if row.get("is_unintelligible") or not (row.get("text") or "").strip():
            return "unintelligible"
        if (row.get("source_collection") or "").lower() in TEST_LIKE_COLLECTIONS:
            return "test_collection"
        if row.get("language_code") not in self.languages:
            return "language"
        if reason := self._common(row):
            return reason
        conf = row.get("confidence")
        if conf is not None and conf < self.confidence_floor(row.get("provider")):
            return "low_confidence"
        return None
