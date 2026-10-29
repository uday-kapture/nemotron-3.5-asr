import audioop
import io

import numpy as np
import pytest
import soundfile as sf

from hinglish_asr.audio import AudioError, convert_audio


def _pcm(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(0, 3000, n), -32768, 32767).astype("<i2")


def _wav(path, pcm, rate, channels=1):
    data = pcm if channels == 1 else np.stack([pcm] * channels, axis=1)
    sf.write(path, data, rate, subtype="PCM_16", format="WAV")
    return path


def test_16k_input_is_copied_sample_for_sample(tmp_path):
    pcm = _pcm(16000 * 2)
    info = convert_audio(_wav(tmp_path / "in.wav", pcm, 16000), tmp_path / "out.wav")
    out, rate = sf.read(tmp_path / "out.wav", dtype="int16")
    assert rate == 16000
    np.testing.assert_array_equal(out, pcm)
    assert info.orig_sr == 16000
    assert info.duration == pytest.approx(2.0)


def test_8k_input_matches_productions_audioop_resampler(tmp_path):
    # Production (LITranscriber) upsamples with audioop.ratecv, state carried
    # across packets; one call over the whole buffer gives the same bytes.
    pcm = _pcm(8000 * 3 + 37)
    expected, _ = audioop.ratecv(pcm.tobytes(), 2, 1, 8000, 16000, None)
    convert_audio(_wav(tmp_path / "in.wav", pcm, 8000), tmp_path / "out.wav")
    out, rate = sf.read(tmp_path / "out.wav", dtype="int16")
    assert rate == 16000
    np.testing.assert_array_equal(out, np.frombuffer(expected, dtype="<i2"))


def test_output_is_16bit_pcm_wav(tmp_path):
    convert_audio(_wav(tmp_path / "in.wav", _pcm(8000), 8000), tmp_path / "out.wav")
    info = sf.info(tmp_path / "out.wav")
    assert (info.format, info.subtype, info.channels) == ("WAV", "PCM_16", 1)


def test_stereo_is_rejected_rather_than_silently_downmixed(tmp_path):
    with pytest.raises(AudioError, match="2 channels"):
        convert_audio(_wav(tmp_path / "in.wav", _pcm(8000), 8000, channels=2), tmp_path / "out.wav")
    assert not (tmp_path / "out.wav").exists()


def test_unsupported_sample_rate_is_rejected(tmp_path):
    with pytest.raises(AudioError, match="44100"):
        convert_audio(_wav(tmp_path / "in.wav", _pcm(44100), 44100), tmp_path / "out.wav")


def test_offset_and_duration_cut_exact_samples(tmp_path):
    pcm = _pcm(16000 * 3)
    info = convert_audio(_wav(tmp_path / "in.wav", pcm, 16000), tmp_path / "out.wav", offset=1.0, duration=0.5)
    out, _ = sf.read(tmp_path / "out.wav", dtype="int16")
    np.testing.assert_array_equal(out, pcm[16000:24000])
    assert info.duration == pytest.approx(0.5)


def test_offset_on_8k_source_is_applied_before_upsampling(tmp_path):
    pcm = _pcm(8000 * 3)
    info = convert_audio(_wav(tmp_path / "in.wav", pcm, 8000), tmp_path / "out.wav", offset=2.0)
    assert info.duration == pytest.approx(1.0, abs=1e-3)


def test_audio_given_as_bytes_is_accepted(tmp_path):
    buf = io.BytesIO()
    pcm = _pcm(16000)
    sf.write(buf, pcm, 16000, subtype="PCM_16", format="WAV")
    convert_audio(buf.getvalue(), tmp_path / "out.wav")
    out, _ = sf.read(tmp_path / "out.wav", dtype="int16")
    np.testing.assert_array_equal(out, pcm)
