"""MLflow for every stage: one place that checks the setup and opens runs.

Three traps of NeMo's own MLflow hookup this avoids:
- exp_manager's MLflow logger breaks on `run_name` (NeMo PR #15261), so
  training uses Lightning's MLFlowLogger directly (`lightning_logger`);
- with no tracking URI, runs land silently in a local store on whichever box
  ran them, so a missing MLFLOW_TRACKING_URI is an error unless local runs are
  asked for (then: ./mlflow.db, as MLflow 3 no longer accepts file stores);
- with both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN set, MLflow
  sends Basic auth only.

Runs carry a `stage` tag (baseline, pilot, sweep, train, benchmark). A
training run's id is written to <run_dir>/mlflow_run_id so the benchmark
nests one run per candidate checkpoint under it. Weights are never logged
automatically (log_model=False): only the chosen model is uploaded.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PARENT_TAG = "mlflow.parentRunId"
MAX_PARAM_CHARS = 500
PARAM_NAME_REFUSED = re.compile(r"[^A-Za-z0-9_\-. :/]")


class TrackingError(RuntimeError):
    pass


def tracking_uri(allow_local: bool = False) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_TOKEN"):
        print("warning: both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_TOKEN are set; "
              "MLflow will use Basic auth only", file=sys.stderr)
    if uri:
        return uri
    if allow_local:
        return f"sqlite:///{Path('mlflow.db').resolve()}"
    raise TrackingError("MLFLOW_TRACKING_URI is not set; runs would land on this box only. Set it, or allow local runs.")


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Nested config -> {"a.b": "value"}, values cut to MLflow's param limit.

    Characters MLflow refuses in names become "_" (att_context_probs keys
    like "56,0" -> "56_0").
    """
    out = {}
    for k, v in d.items():
        key = f"{prefix}{PARAM_NAME_REFUSED.sub('_', str(k))}"
        if isinstance(v, dict):
            out.update(flatten(v, f"{key}."))
        else:
            text = v if isinstance(v, str) else json.dumps(v)
            out[key] = text[:MAX_PARAM_CHARS]
    return out


def numeric(metrics: dict, prefix: str = "") -> dict[str, float]:
    """The finite numbers in `metrics` (None and non-numbers dropped), prefixed."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, int | float) or v != v:
            continue
        out[f"{prefix}{k}"] = float(v)
    return out


@contextmanager
def run(
    experiment: str,
    run_name: str | None = None,
    *,
    stage: str,
    parent_run_id: str | None = None,
    run_id: str | None = None,
    tags: dict | None = None,
    allow_local: bool = False,
) -> Iterator:
    """Start (or with `run_id`, reopen) a run; `parent_run_id` nests it in the UI."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(allow_local))
    mlflow.set_experiment(experiment)
    all_tags = {"stage": stage, **(tags or {})}
    if parent_run_id:
        all_tags[PARENT_TAG] = parent_run_id
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else run_name) as active:
        mlflow.set_tags(all_tags)
        yield active


def log_params(params: dict) -> None:
    import mlflow

    flat = flatten(params)
    items = list(flat.items())
    for i in range(0, len(items), 100):  # MLflow's batch limit
        mlflow.log_params(dict(items[i : i + 100]))


def log_metrics(metrics: dict, prefix: str = "", step: int | None = None) -> None:
    import mlflow

    values = numeric(metrics, prefix)
    if values:
        mlflow.log_metrics(values, step=step)


def lightning_logger(
    experiment: str,
    run_name: str,
    *,
    tags: dict | None = None,
    run_id: str | None = None,
    allow_local: bool = False,
):
    from lightning.pytorch.loggers import MLFlowLogger

    return MLFlowLogger(
        experiment_name=experiment,
        run_name=run_name,
        tracking_uri=tracking_uri(allow_local),
        tags=tags,
        run_id=run_id,
        log_model=False,
    )
