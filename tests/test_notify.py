import io
import json
import urllib.error
from urllib.parse import parse_qs, urlsplit

import pytest

from hinglish_asr import notify
from hinglish_asr.train import pilot_text, sweep_text

URL = "https://chat.googleapis.com/v1/spaces/AAA/messages?key=secret-key&token=secret-token"


class FakeOpener:
    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []

    def __call__(self, request, timeout):
        self.requests.append(request)
        outcome = self.outcomes.pop(0) if self.outcomes else 200
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome != 200:
            raise urllib.error.HTTPError(request.full_url, outcome, "error", {}, io.BytesIO())
        return io.BytesIO(b"{}")


def _chat(*outcomes, thread="calls-v1"):
    opener = FakeOpener(*outcomes)
    return notify.GChat(URL, thread, opener=opener, sleep=lambda s: None), opener


def test_posts_text_into_the_run_thread():
    chat, opener = _chat()
    assert chat.send("hello")
    (req,) = opener.requests
    query = parse_qs(urlsplit(req.full_url).query)
    assert query["key"] == ["secret-key"] and query["threadKey"] == ["calls-v1"]
    assert query["messageReplyOption"] == ["REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"]
    assert json.loads(req.data) == {"text": "hello"}


def test_retries_rate_limits_and_server_errors_then_gives_up_quietly(capsys):
    chat, opener = _chat(429, 503)
    assert chat.send("x") and len(opener.requests) == 3
    chat, opener = _chat(500, 500, 500)
    assert not chat.send("x") and len(opener.requests) == 3
    chat, opener = _chat(400)  # a bad request is not retried
    assert not chat.send("x") and len(opener.requests) == 1
    chat, opener = _chat(TimeoutError(), TimeoutError(), TimeoutError())
    assert not chat.send("x")
    assert "secret" not in capsys.readouterr().err  # the URL holds the key


def test_without_a_webhook_nothing_is_sent(monkeypatch):
    monkeypatch.delenv(notify.WEBHOOK_ENV, raising=False)
    chat = notify.GChat.from_env("run")
    assert not chat.enabled and not chat.send("x")


def test_long_messages_are_cut():
    chat, opener = _chat()
    chat.send("x" * 10_000)
    assert len(json.loads(opener.requests[0].data)["text"]) <= notify.MAX_CHARS


def test_message_text():
    assert notify.duration(45) == "45s" and notify.duration(125) == "2m" and notify.duration(3 * 3600 + 300) == "3h 05m"
    assert notify.progress_text("r", 1500, 50000, 0.41234, 1.2e-4, 0.43, 1160.4, 3 * 3600) == (
        "*r* step 1,500 / 50,000 (3%) · loss 0.4123 · lr 1.20e-04 · 0.43 s/step · 1,160× realtime · ETA 3h 00m"
    )
    assert notify.epoch_text("r", 3, 50, 3000, 0.181, 0.35, 0.19) == (
        "*r* epoch 3 / 50 done (step 3,000) · val_wer 18.10% (new best) · val_loss 0.35"
    )
    assert "(best 17.00%)" in notify.epoch_text("r", 4, 50, 4000, 0.18, None, 0.17)
    assert "resumed at step 12,000" in notify.started_text("r", "train", "cosine", 12000, 50000, 500, "H200")


def test_failure_text_keeps_the_end_of_the_traceback():
    def boom():
        raise RuntimeError("CUDA out of memory")

    try:
        boom()
    except RuntimeError as e:
        text = notify.failed_text("r", e, 2350)
    assert text.startswith("🔴 *r* failed at step 2,350: RuntimeError") and "CUDA out of memory" in text
    assert notify.failed_text("r", KeyboardInterrupt(), None) == "⏹️ *r* stopped before training started (KeyboardInterrupt)"


def test_driver_summaries():
    pilot = {
        "results": [
            {"batch_duration": 400, "status": "ok", "audio_s_per_s": 1100.0, "s_per_step": 0.36, "peak_reserved_gb": 80.2},
            {"batch_duration": 800, "status": "oom"},
        ],
        "chosen": {"batch_duration": 400}, "max_steps": 62100, "est_hours": 6.2, "passes": 3.0,
    }
    assert pilot_text(pilot).splitlines()[1:] == [
        "• batch 400 s: 1,100× realtime, 0.36 s/step, peak 80 GB",
        "• batch 800 s: oom",
        "chosen: batch 400 s → max_steps 62,100, ~6.2 h for 3 passes",
    ]
    sweep = {"results": [{"arm": "noam", "best_val_wer": 0.19}, {"name": "sweep-cosine", "status": "failed"}],
             "winner": "noam"}
    assert sweep_text("sweep", sweep).splitlines()[1:] == [
        "• noam: best val_wer 19.00%", "• sweep-cosine: failed", "winner: noam",
    ]


def test_callback_in_a_real_lightning_loop(tmp_path):
    pl = pytest.importorskip("lightning.pytorch")
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from hinglish_asr.train import _notify_callback

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))
            self.wers = iter([0.30, 0.20, 0.25])

        def training_step(self, batch, batch_idx):
            loss = (self.w * batch[0]).mean()
            if self.global_step == 3:
                loss = loss * float("nan")
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            pass

        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking:
                self.log("val_wer", next(self.wers))
                self.log("val_loss", 1.0)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    class Recorder:
        def __init__(self):
            self.sent = []

        def send(self, text):
            self.sent.append(text)
            return True

    data = TensorDataset(torch.ones(64, 4), torch.full((64,), 16000))  # batch[1]: audio lengths, 1 s each
    loader = lambda: DataLoader(data, batch_size=2)  # noqa: E731
    chat = Recorder()
    cfg = {"arm": "cosine", "model": {"train_ds": {"batch_duration": 2}}, "notify": {"every_n_steps": 2}}
    callback = _notify_callback(chat, "tiny", "train", cfg, check_loss_every=1)
    trainer = pl.Trainer(
        max_steps=9, limit_train_batches=3, limit_val_batches=1, accelerator="cpu", logger=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=tmp_path,
        callbacks=[callback, pl.callbacks.ModelCheckpoint(dirpath=tmp_path, monitor="val_wer", save_top_k=1)],
    )
    trainer.fit(Tiny(), loader(), loader())

    assert chat.sent[0].startswith("*tiny* started (train, arm cosine): 9 steps")
    assert [m.split(" · ")[0] for m in chat.sent if " step " in m and "epoch" not in m and "⚠️" not in m] == [
        "*tiny* step 2 / 9 (22%)", "*tiny* step 4 / 9 (44%)", "*tiny* step 6 / 9 (67%)", "*tiny* step 8 / 9 (89%)",
    ]
    assert "realtime" in chat.sent[1] and "ETA" in chat.sent[1]
    assert sum("loss is nan at step 4" in m for m in chat.sent) == 1
    epochs = [m for m in chat.sent if "epoch" in m]
    assert [e.split(" · ")[0] for e in epochs] == [
        "*tiny* epoch 1 / 3 done (step 3)", "*tiny* epoch 2 / 3 done (step 6)", "*tiny* epoch 3 / 3 done (step 9)",
    ]
    assert "val_wer 30.00% (new best)" in epochs[0] and "val_wer 20.00% (new best)" in epochs[1]
    assert "val_wer 25.00% (best 20.00%)" in epochs[2]
    assert callback.step == 9


def test_fit_reports_failure_and_finish(monkeypatch):
    pytest.importorskip("lightning.pytorch")
    from hinglish_asr import train

    sent = []
    monkeypatch.setenv(notify.WEBHOOK_ENV, URL)
    monkeypatch.setattr(notify.GChat, "send", lambda self, text: sent.append((self.thread, text)) or True)
    cfg = {"model": {"train_ds": {}}, "notify": {"every_n_steps": 500}}

    def crash(*args):
        raise FileNotFoundError("configs/input_cfg.yaml")

    monkeypatch.setattr(train, "_fit", crash)
    with pytest.raises(FileNotFoundError):
        train.fit(cfg, "calls-v1", "train")
    assert sent[-1][0] == "calls-v1"
    assert sent[-1][1].startswith("🔴 *calls-v1* failed before training started: FileNotFoundError")

    monkeypatch.setattr(train, "_fit", lambda *args: {"best_val_wer": 0.1734})
    train.fit(cfg, "calls-v1", "train")
    assert sent[-1][1].startswith("✅ *calls-v1* finished, best val_wer 17.34% in ")

    sent.clear()
    train.fit(cfg, "pilot-bd400", "pilot")  # pilot runs stay quiet
    assert sent == []
