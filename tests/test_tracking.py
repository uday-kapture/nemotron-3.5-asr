import pytest

from hinglish_asr import tracking


def test_missing_tracking_uri_is_an_error_unless_local_is_allowed(monkeypatch, tmp_path):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(tracking.TrackingError, match="MLFLOW_TRACKING_URI"):
        tracking.tracking_uri()
    monkeypatch.chdir(tmp_path)
    assert tracking.tracking_uri(allow_local=True) == f"sqlite:///{(tmp_path / 'mlflow.db').resolve()}"


def test_basic_and_token_auth_together_warn(monkeypatch, capsys):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "u")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "t")
    assert tracking.tracking_uri() == "https://mlflow.example"
    assert "Basic auth only" in capsys.readouterr().err


def test_flatten_makes_names_mlflow_accepts():
    flat = tracking.flatten({"model": {"att_context_probs": {"56,0": 0.55, "56,13": 0.15}}})
    assert flat == {"model.att_context_probs.56_0": "0.55", "model.att_context_probs.56_13": "0.15"}


def test_flatten_nests_keys_and_truncates():
    flat = tracking.flatten({"a": {"b": 1, "c": [56, 0]}, "s": "x" * 600, "n": None})
    assert flat["a.b"] == "1" and flat["a.c"] == "[56, 0]" and flat["n"] == "null"
    assert len(flat["s"]) == tracking.MAX_PARAM_CHARS


def test_numeric_keeps_finite_numbers_only():
    assert tracking.numeric({"wer": 0.1, "n": 3, "none": None, "nan": float("nan"), "flag": True, "s": "x"}, "val/") == {
        "val/wer": 0.1,
        "val/n": 3.0,
    }


def test_runs_nest_under_a_parent(monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    with tracking.run("exp", "train", stage="train") as parent:
        tracking.log_params({"model": {"att_context_size": [56, 0]}})
    with tracking.run("exp", "cand", stage="benchmark", parent_run_id=parent.info.run_id):
        tracking.log_metrics({"wer": 0.2, "skip": None}, prefix="val/")
    runs = mlflow.search_runs(experiment_names=["exp"], output_format="list")
    child = next(r for r in runs if r.data.tags["stage"] == "benchmark")
    assert child.data.tags[tracking.PARENT_TAG] == parent.info.run_id
    assert child.data.metrics == {"val/wer": 0.2}
    assert next(r for r in runs if r.data.tags["stage"] == "train").data.params == {"model.att_context_size": "[56, 0]"}
