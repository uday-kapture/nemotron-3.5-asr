"""Where utterances come from: local JSONL manifests or parquet (local or on the HF Hub).

Parquet is read one row group at a time and only the row groups that hold
wanted rows are fetched, so sampling a few hundred rows from a 90 GB dataset
downloads a few GB, not the whole file.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .split import group_key


class SourceError(ValueError):
    """Input that cannot be read as a source of utterances."""


@dataclass(frozen=True)
class Record:
    id: str
    audio: str | bytes | None  # file path, encoded audio bytes, or None when already converted
    text: str
    group: str  # speaker/call; utterances sharing it land in the same split
    offset: float = 0.0
    duration: float | None = None
    row: int | None = None  # original parquet row index


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.]+", "_", name)


def read_jsonl_source(path: str | Path, group_pattern: str | None = None) -> Iterator[Record]:
    """Rows: audio_filepath, text, and optionally id, offset, duration, group.

    Relative audio paths resolve against the manifest's directory. Without a
    `group`, utterances group by audio file (all segments of a call together),
    or by `group_pattern`'s first capture group on the file name.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                audio, text = row["audio_filepath"], row["text"]
            except (ValueError, KeyError) as exc:
                raise SourceError(f"{path} line {lineno}: {exc!r}") from exc
            audio_path = Path(audio)
            if not audio_path.is_absolute():
                audio_path = path.parent / audio_path
            audio = str(audio_path)
            offset = float(row.get("offset", 0.0))
            duration = float(row["duration"]) if row.get("duration") is not None else None
            rid = row.get("id")
            if rid is None:
                rid = f"{audio_path.stem}_{hashlib.sha1(audio.encode()).hexdigest()[:8]}"
                if offset or duration is not None:
                    rid += f"_{round(offset * 1000)}_{round((duration or 0) * 1000)}"
            group = row.get("group") or group_key(audio, group_pattern, fallback=audio)
            yield Record(id=_safe(str(rid)), audio=audio, text=text, group=str(group), offset=offset, duration=duration)


class ParquetSource:
    def __init__(self, files: list[str], fs=None):
        self.files = list(files)
        self.fs = fs  # fsspec filesystem (e.g. HfFileSystem); None for local paths

    @classmethod
    def from_hf(cls, repo: str, revision: str, split: str) -> "ParquetSource":
        """The parquet file(s) of `split` in an HF dataset repo, pinned to `revision`."""
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(f"datasets/{repo}@{revision}/**/*.parquet"))
        if len(files) > 1:
            files = [f for f in files if split in f.split(f"@{revision}/", 1)[-1]]
        if not files:
            raise SourceError(f"no parquet files for split {split!r} in {repo}@{revision}")
        return cls(files, fs)

    def _open(self, path: str):
        if self.fs is None:
            return open(path, "rb")
        return self.fs.open(path, "rb", block_size=8 << 20)

    def row_group_bounds(self) -> list[tuple[int, int]]:
        bounds, offset = [], 0
        for path in self.files:
            with self._open(path) as f:
                md = pq.ParquetFile(f).metadata
                for g in range(md.num_row_groups):
                    n = md.row_group(g).num_rows
                    bounds.append((offset, offset + n))
                    offset += n
        return bounds

    def records(
        self,
        rows: Iterable[int] | None = None,
        exclude: Iterable[int] = (),
        audio_column: str = "audio",
        text_column: str = "text",
        group_pattern: str | None = None,
        id_prefix: str = "row",
        references: dict[int, dict] | None = None,
        converted: Callable[[str], bool] | None = None,
    ) -> Iterator[Record]:
        """Records in row order; `references` ({row: {id, reference}}) override id and text.

        `converted(id)` marks rows whose audio an earlier run already wrote. A
        row group whose remaining rows are all converted is read without its
        audio bytes (path and text only), and those records carry audio=None.
        """
        wanted = sorted(set(rows)) if rows is not None else None
        exclude = set(exclude)

        def rid_of(row: int) -> str:
            if references and row in references:
                return _safe(references[row]["id"])
            return _safe(f"{id_prefix}_{row}")

        offset = 0
        for path in self.files:
            with self._open(path) as f:
                pf = pq.ParquetFile(f)
                for g in range(pf.metadata.num_row_groups):
                    start = offset
                    offset += pf.metadata.row_group(g).num_rows
                    todo = [
                        r
                        for r in range(start, offset)
                        if r not in exclude and (wanted is None or _contains(wanted, r))
                    ]
                    if not todo:
                        continue
                    skip_audio = converted is not None and all(converted(rid_of(r)) for r in todo)
                    audio_col = f"{audio_column}.path" if skip_audio else audio_column
                    table = pf.read_row_group(g, columns=[audio_col, text_column])
                    audios = table.column(audio_column).to_pylist()
                    texts = table.column(text_column).to_pylist()
                    for row in todo:
                        audio, text = audios[row - start], texts[row - start]
                        rid, ref = rid_of(row), text
                        if references and row in references:
                            ref = references[row]["reference"]
                        name = (audio or {}).get("path") or ""
                        data = None if skip_audio else ((audio or {}).get("bytes") or name)
                        yield Record(
                            id=rid,
                            audio=data,
                            text=ref or "",
                            group=group_key(name, group_pattern, fallback=rid),
                            row=row,
                        )


def _contains(sorted_list: list[int], x: int) -> bool:
    i = bisect.bisect_left(sorted_list, x)
    return i < len(sorted_list) and sorted_list[i] == x


def read_hush_eval_manifest(path: str | Path) -> tuple[dict, list[dict]]:
    """(source, samples) from a Hush-Eval frozen evaluation list (version 1)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, list):
        doc = {"version": 1, "samples": doc}
    if doc.get("version") != 1:
        raise SourceError(f"{path}: unsupported Hush-Eval manifest version {doc.get('version')!r}")
    return doc.get("source", {"type": "local"}), doc["samples"]
