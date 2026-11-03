from types import SimpleNamespace

import pytest

from hinglish_asr.hub import HubCheckpoints, ensure_private_repo, rank_checkpoints


class FakeModelRepo:
    def __init__(self, private=True, files=()):
        self.private = private
        self.files = set(files)
        self.commits = []

    def create_repo(self, repo, repo_type, private, exist_ok):
        assert private

    def repo_info(self, repo, repo_type):
        return SimpleNamespace(private=self.private, sha="sha")

    def list_repo_files(self, repo, repo_type):
        return sorted(self.files)

    def create_commit(self, repo, operations, repo_type, commit_message):
        for op in operations:
            if type(op).__name__ == "CommitOperationAdd":
                self.files.add(op.path_in_repo)
            else:
                self.files.discard(op.path_in_repo)
        self.commits.append(commit_message)
        return SimpleNamespace(oid="c")


def _ck(step, wer):
    return f"step={step}-val_wer={wer:.4f}.ckpt"


def test_rank_checkpoints_best_first_without_last():
    names = [_ck(1000, 0.3), _ck(2000, 0.2), "step=2000-val_wer=0.2000-last.ckpt", "notes.txt"]
    assert rank_checkpoints(names) == [(_ck(2000, 0.2), 0.2), (_ck(1000, 0.3), 0.3)]


def test_hub_keeps_only_the_k_best(tmp_path):
    hub_api = FakeModelRepo()
    hub = HubCheckpoints("org/ckpts", "calls-v1", keep=2, api=hub_api)
    local = tmp_path / "w.ckpt"
    local.write_bytes(b"weights")
    assert hub.add(local, _ck(1000, 0.30))
    assert hub.add(local, _ck(2000, 0.25))
    assert hub.add(local, _ck(3000, 0.20))  # pushes 0.30 out in the same commit
    assert not hub.add(local, _ck(4000, 0.40))  # worse than both: not uploaded
    assert [n for n, _ in hub.list()] == [_ck(3000, 0.20), _ck(2000, 0.25)]
    assert hub_api.files == {f"runs/calls-v1/{_ck(3000, 0.20)}", f"runs/calls-v1/{_ck(2000, 0.25)}"}
    assert len(hub_api.commits) == 3


def test_runs_do_not_see_each_others_checkpoints():
    hub_api = FakeModelRepo(files=[f"runs/other/{_ck(1, 0.1)}", f"runs/calls-v1/{_ck(2, 0.2)}"])
    assert HubCheckpoints("org/ckpts", "calls-v1", api=hub_api).list() == [(_ck(2, 0.2), 0.2)]


def test_private_repo_checks():
    with pytest.raises(ValueError, match="public"):
        ensure_private_repo(FakeModelRepo(private=False), "org/x", "model")
    with pytest.raises(ValueError, match="refusing"):
        ensure_private_repo(FakeModelRepo(), "org/src", "dataset", forbidden=("org/src",))


def test_reuploading_a_name_does_not_take_two_top_k_slots(tmp_path):
    hub_api = FakeModelRepo()
    hub = HubCheckpoints("org/ckpts", "calls-v1", keep=2, api=hub_api)
    local = tmp_path / "w.ckpt"
    local.write_bytes(b"weights")
    hub.add(local, _ck(1000, 0.30))
    hub.add(local, _ck(2000, 0.25))
    assert hub.add(local, _ck(2000, 0.25))  # the same validation again, after a resume
    assert [n for n, _ in hub.list()] == [_ck(2000, 0.25), _ck(1000, 0.30)]
