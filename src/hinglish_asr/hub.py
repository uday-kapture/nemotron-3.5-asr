"""Private Hugging Face repos for what does not fit on the training box.

Training shards (`convert --upload-to`) and top-k checkpoints (`train` with
`checkpoint_hub_repo`) both go to private repos: they hold call audio, or
weights trained on it. Every repo is created private and checked to be
private before anything is uploaded.

Checkpoints of run <name> live under runs/<name>/ in a model repo, as weights
only ({"state_dict", "global_step"}, ~2.4 GB instead of ~8 GB with optimizer
state). Each upload is one commit that also deletes whatever falls out of the
top k, so the repo always holds the k best by the val_wer in the file name.
`benchmark --push-winner` adds the chosen model and its report next to them:
runs/<name>/winner.nemo and runs/<name>/benchmark/benchmark.{md,json}.
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete

VAL_WER = re.compile(r"val_wer=([0-9]+\.[0-9]+)")


def ensure_private_repo(api, repo: str, repo_type: str, forbidden: tuple[str, ...] = ()) -> None:
    if repo in forbidden:
        raise ValueError(f"refusing to upload into {repo}")
    api.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    if not api.repo_info(repo, repo_type=repo_type).private:
        raise ValueError(f"{repo} is public; call audio and weights trained on it go only to a private repo")


def rank_checkpoints(names: list[str]) -> list[tuple[str, float]]:
    """(name, val_wer) best first, for names carrying val_wer; -last checkpoints excluded."""
    ranked = []
    for name in names:
        stem = Path(name).stem
        if stem.endswith("-last"):
            continue
        if m := VAL_WER.search(stem):
            ranked.append((name, float(m.group(1))))
    return sorted(ranked, key=lambda t: (t[1], t[0]))


class HubCheckpoints:
    """The top-k weights of one training run in a private HF model repo."""

    def __init__(self, repo: str, run: str, keep: int = 5, api=None, check: bool = True):
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        if check:
            ensure_private_repo(api, repo, "model")
        self.api, self.repo, self.run, self.keep = api, repo, run, keep
        self.prefix = f"runs/{run}"

    def list(self) -> list[tuple[str, float]]:
        files = self.api.list_repo_files(self.repo, repo_type="model")
        return rank_checkpoints([f.rsplit("/", 1)[-1] for f in files if f.startswith(f"{self.prefix}/")])

    def add(self, local: str | Path, name: str) -> bool:
        """Upload `local` as `name` if it ranks in the top k; drop what falls out. False if not uploaded."""
        existing = self.list()
        # A name already there (validation repeated at that step after a resume) is replaced, not counted twice.
        names = list(dict.fromkeys([n for n, _ in existing] + [name]))
        keep = {n for n, _ in rank_checkpoints(names)[: self.keep]}
        if name not in keep:
            return False
        ops = [CommitOperationAdd(f"{self.prefix}/{name}", str(local))]
        ops += [CommitOperationDelete(f"{self.prefix}/{n}") for n, _ in existing if n not in keep]
        self.api.create_commit(self.repo, operations=ops, repo_type="model", commit_message=f"{self.run}: {name}")
        return True

    def push(self, files: dict[str, str | Path], message: str) -> None:
        """Upload other files of the run (the benchmark's winner.nemo, report) under runs/<run>/, one commit.

        Their names carry no val_wer, so `list` and `add` never rank or delete them.
        """
        ops = [CommitOperationAdd(f"{self.prefix}/{name}", str(path)) for name, path in files.items()]
        self.api.create_commit(self.repo, operations=ops, repo_type="model", commit_message=message)

    def download(self, name: str, cache_dir: str | Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(self.repo, f"{self.prefix}/{name}", repo_type="model", local_dir=cache_dir))
