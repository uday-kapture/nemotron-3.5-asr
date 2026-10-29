"""Convert source audio to the 16 kHz mono PCM_16 WAV the model expects.

8 kHz telephony audio is upsampled with audioop.ratecv in 20 ms packets with
state carried, exactly as production's LITranscriber.send_audio does (see
Hush-Eval's telephony.py), so the model is trained and evaluated on the same
bytes it will hear live. NVIDIA recommends resampling offline rather than in
the dataloader, which is "very time consuming".
"""

from __future__ import annotations

import io
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

with warnings.catch_warnings():  # deprecated in 3.11, removed in 3.13; production uses it
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop

SR = 16000
SOURCE_RATES = (8000, SR)
PACKET_SECONDS = 0.02


class AudioError(ValueError):
    """Audio that cannot be converted the way production would."""


@dataclass(frozen=True)
class AudioInfo:
    duration: float  # seconds, of the written 16 kHz audio
    orig_sr: int


def to_16k(pcm: np.ndarray, rate: int) -> np.ndarray:
    if rate == SR:
        return pcm
    data = np.asarray(pcm, dtype="<i2").tobytes()
    step = 2 * round(rate * PACKET_SECONDS)
    state, out = None, []
    for i in range(0, len(data), step):
        chunk, state = audioop.ratecv(data[i : i + step], 2, 1, rate, SR, state)
        out.append(chunk)
    return np.frombuffer(b"".join(out), dtype="<i2")


def convert_audio(
    src: str | Path | bytes,
    dst: str | Path,
    offset: float = 0.0,
    duration: float | None = None,
) -> AudioInfo:
    """Write `src[offset:offset+duration]` to `dst` as 16 kHz mono PCM_16 WAV."""
    with sf.SoundFile(io.BytesIO(src) if isinstance(src, bytes) else src) as f:
        if f.channels != 1:
            raise AudioError(f"{f.channels} channels; need mono")
        if f.samplerate not in SOURCE_RATES:
            raise AudioError(f"{f.samplerate} Hz; need one of {SOURCE_RATES}")
        rate = f.samplerate
        if offset:
            f.seek(int(round(offset * rate)))
        frames = int(round(duration * rate)) if duration is not None else -1
        pcm = f.read(frames, dtype="int16")

    out = to_16k(pcm, rate)
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    sf.write(tmp, out, SR, subtype="PCM_16", format="WAV")
    os.replace(tmp, dst)
    return AudioInfo(duration=len(out) / SR, orig_sr=rate)
