import pytest

from hinglish_asr.benchmark import breakdown, choose, duration_bucket, evaluate, find_checkpoints, render_markdown


def test_find_checkpoints_orders_topk_by_val_wer_and_finds_last(tmp_path):
    for name in ("step=2000-val_wer=0.2100.ckpt", "step=4000-val_wer=0.1900.ckpt", "step=4000-val_wer=0.1900-last.ckpt",
                 "notes.txt"):
        (tmp_path / name).write_text("")
    topk, last = find_checkpoints(tmp_path)
    assert [(p.name, w) for p, w in topk] == [("step=4000-val_wer=0.1900.ckpt", 0.19), ("step=2000-val_wer=0.2100.ckpt", 0.21)]
    assert last.name == "step=4000-val_wer=0.1900-last.ckpt"


@pytest.mark.parametrize("d,bucket", [(1.5, "<2s"), (2.0, "2-5s"), (5.0, "2-5s"), (5.1, ">5s")])
def test_duration_bucket(d, bucket):
    assert duration_bucket(d) == bucket


def test_breakdown_keeps_top_values_and_pools_the_rest():
    rows = [{"client_id": c, "duration": d} for c, d in [("a", 3600), ("a", 3600), ("b", 1800), ("c", 60)]]
    per = [(1, 10), (0, 10), (5, 10), None]
    out = breakdown(rows, per, lambda r: r["client_id"], top=2)
    assert set(out) == {"a", "b", "other"}
    assert out["a"]["wer"] == pytest.approx(0.05)
    assert out["other"]["wer"] is None and out["other"]["hours"] == pytest.approx(60 / 3600, abs=1e-3)


def _rows():
    return [
        {"text": "मेरा order cancel करो", "duration": 1.5, "language": "hi", "channel": "right", "client_id": "c1", "group": "g1"},
        {"text": "पैसे नहीं आए", "duration": 3.0, "language": "hi", "channel": "left", "client_id": "c2", "group": "g2"},
    ]


def test_evaluate_reports_plain_lenient_and_breakdowns():
    scores, per = evaluate(_rows(), ["मेरा ऑर्डर कैंसल करो", "पैसे आए"])
    assert scores["wer"] == pytest.approx(3 / 7)
    assert scores["lenient_wer"] == pytest.approx(1 / 7)
    assert per == [(2, 4), (1, 3)]
    assert scores["by"]["channel"]["left"]["wer"] == pytest.approx(1 / 3)
    assert "semdist_mean" not in scores


def test_choose_and_markdown():
    v = lambda wer, len_: {"wer": wer, "cer": 0.1, "lenient_wer": len_, "wer_devanagari": None, "wer_latin": None}
    cands = {
        "base": {"val": v(0.30, 0.20), "test": v(0.31, 0.21)},
        "top1": {"val": v(0.20, 0.15), "test": v(0.22, 0.16), "val_vs_base": {"delta": -0.1, "ci95": [-0.12, -0.08]}},
        "avg-top2": {"val": v(0.21, 0.14), "val_vs_base": {"delta": -0.09, "ci95": [-0.11, -0.07]}},
    }
    assert choose(cands) == "top1" and choose(cands, "lenient_wer") == "avg-top2"
    meta = {"run": "runs/x", "latency_ms": 80, "att_context": [56, 0], "prompt": "auto", "val": "v", "test": "t",
            "val_utterances": 2, "val_hours": 1.0, "test_utterances": 2, "test_hours": 1.0}
    md = render_markdown({"meta": meta, "winner": "top1", "candidates": cands,
                          "checks": {"beats base on test": {"pass": True, "detail": "ok"}}})
    assert "| top1 **(winner)** | 20.00% | -10.00 [-12.00, -8.00]" in md
    assert "## Test (reported once)" in md and "| avg-top2 |" in md.split("## Test")[0]
    assert "Held-out clients" not in md and "optimistic for the agent side" in md


def test_markdown_reports_held_out_clients_when_given():
    v = {"wer": 0.2, "cer": 0.1, "lenient_wer": 0.15}
    cands = {"base": {"val": v, "test": v, "client_test": {**v, "wer": 0.4}},
             "top1": {"val": v, "test": v, "client_test": v,
                      "client_test_vs_base": {"delta": -0.2, "ci95": [-0.25, -0.15]}}}
    meta = {"run": "runs/x", "latency_ms": 80, "att_context": [56, 0], "prompt": "auto", "val": "v", "test": "t",
            "client_test": "ct", "val_utterances": 2, "val_hours": 1.0, "test_utterances": 2, "test_hours": 1.0,
            "client_test_utterances": 3, "client_test_hours": 0.5}
    md = render_markdown({"meta": meta, "winner": "top1", "candidates": cands, "checks": {}})
    section = md.split("## Held-out clients (reported once)")[1]
    assert "| base | 40.00% |" in section and "| top1 | 20.00% | -20.00 [-25.00, -15.00]" in section
    assert "client_test: `ct` (3 utterances, 0.50 h)" in md
