"""Google Chat messages about training runs, through a space's incoming webhook.

The webhook URL is a secret (anyone holding it can post into the space), so it
comes from the GCHAT_WEBHOOK_URL environment variable, never from a config
file; without it nothing is sent. Each run posts into its own thread
(threadKey = run name), so a resumed run continues the thread it started.

Messages carry run names, step counts and metrics, never transcripts. Sending
never raises: a failed post is reported on stderr and training goes on.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from urllib.parse import urlencode

WEBHOOK_ENV = "GCHAT_WEBHOOK_URL"
MAX_CHARS = 4000  # Chat takes far more; longer is unreadable in a thread
TRACEBACK_LINES = 12


class GChat:
    def __init__(self, url: str | None, thread: str | None = None, *, timeout: float = 10.0, retries: int = 2,
                 opener=urllib.request.urlopen, sleep=time.sleep):
        self.url, self.thread, self.timeout, self.retries = url, thread, timeout, retries
        self.opener, self.sleep = opener, sleep

    @classmethod
    def from_env(cls, thread: str | None = None, **kw) -> GChat:
        return cls(os.environ.get(WEBHOOK_ENV) or None, thread, **kw)

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def endpoint(self) -> str:
        if not self.thread:
            return self.url
        reply = {"threadKey": self.thread, "messageReplyOption": "REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD"}
        return f"{self.url}{'&' if '?' in self.url else '?'}{urlencode(reply)}"

    def send(self, text: str) -> bool:
        if not self.url:
            return False
        if len(text) > MAX_CHARS:
            text = text[: MAX_CHARS - 20] + "\n… (cut)"
        request = urllib.request.Request(
            self.endpoint(),
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json; charset=UTF-8"},
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    response.read()
                return True
            except urllib.error.HTTPError as e:  # the URL holds the key: report the status and Google's reason only
                error = f"HTTP {e.code}{_reason(e)}"
                if e.code != 429 and e.code < 500:
                    break
            except OSError as e:  # URLError, timeouts, resets
                error = type(e).__name__
            if attempt < self.retries:
                self.sleep(2 ** (attempt + 1))
        print(f"gchat: message not sent ({error})", file=sys.stderr)
        return False


def _reason(e: urllib.error.HTTPError) -> str:
    try:
        return ": " + json.loads(e.read())["error"]["message"][:200]
    except Exception:
        return ""


# ---------- message text ----------


def duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    h, m = divmod(seconds // 60, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _of(step: int, total: int | None) -> str:
    return f"{step:,} / {total:,}" if total and total > 0 else f"{step:,}"


def started_text(name: str, stage: str, arm: str | None, step: int, max_steps: int, batch_duration, gpu: str) -> str:
    verb = f"resumed at step {step:,}" if step else "started"
    return (f"*{name}* {verb} ({stage}, arm {arm}): {max_steps:,} steps, "
            f"{batch_duration} s of audio per batch, {gpu}")


def progress_text(name: str, step: int, max_steps: int, loss: float | None, lr: float | None,
                  s_per_step: float | None, realtime: float | None, eta_s: float | None) -> str:
    parts = [f"*{name}* step {_of(step, max_steps)}" + (f" ({step / max_steps:.0%})" if max_steps > 0 else "")]
    if loss is not None:
        parts.append(f"loss {loss:.4g}")
    if lr is not None:
        parts.append(f"lr {lr:.2e}")
    if s_per_step:
        parts.append(f"{s_per_step:.2f} s/step")
    if realtime:
        parts.append(f"{realtime:,.0f}× realtime")
    if eta_s is not None:
        parts.append(f"ETA {duration(eta_s)}")
    return " · ".join(parts)


def epoch_text(name: str, epoch: int, epochs: int | None, step: int, val_wer: float | None,
               val_loss: float | None, best_before: float | None) -> str:
    parts = [f"*{name}* epoch {_of(epoch, epochs)} done (step {step:,})"]
    if val_wer is not None:
        if best_before is None or val_wer < best_before:
            parts.append(f"val_wer {val_wer:.2%} (new best)")
        else:
            parts.append(f"val_wer {val_wer:.2%} (best {best_before:.2%})")
    if val_loss is not None:
        parts.append(f"val_loss {val_loss:.4g}")
    return " · ".join(parts)


def bad_loss_text(name: str, step: int, loss: float) -> str:
    return f"⚠️ *{name}* loss is {loss} at step {step:,}"


def finished_text(name: str, step: int | None, best_wer: float | None, elapsed_s: float) -> str:
    best = f", best val_wer {best_wer:.2%}" if best_wer is not None else ""
    at = f" at step {step:,}" if step is not None else ""
    return f"✅ *{name}* finished{at}{best} in {duration(elapsed_s)}"


def failed_text(name: str, exc: BaseException, step: int | None) -> str:
    where = f"at step {step:,}" if step is not None else "before training started"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        return f"⏹️ *{name}* stopped {where} ({type(exc).__name__})"
    tail = "".join(traceback.format_exception(exc)).strip().splitlines()[-TRACEBACK_LINES:]
    return f"🔴 *{name}* failed {where}: {type(exc).__name__}\n```\n" + "\n".join(tail) + "\n```"


def killed_text(name: str, signal_name: str) -> str:
    return f"🔴 *{name}* was killed ({signal_name}); host out of memory or a manual kill"
