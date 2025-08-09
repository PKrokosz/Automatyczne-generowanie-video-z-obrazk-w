import numpy as np
import soundfile as sf
import librosa
import pytest

from ken_burns_reel.audio import extract_beats
from moviepy.audio.AudioClip import AudioClip
try:
    from moviepy.audio.fx.audio_fadein import audio_fadein
    from moviepy.audio.fx.audio_fadeout import audio_fadeout
except ImportError:  # moviepy >=2.0
    from moviepy.audio.fx import AudioFadeIn as audio_fadein, AudioFadeOut as audio_fadeout


def test_extract_beats(tmp_path):
    sr = 22050
    y = librosa.clicks(times=[0.0, 1.0], sr=sr, length=sr * 2)
    path = tmp_path / "clicks.wav"
    sf.write(path, y, sr)
    beats = extract_beats(str(path))
    assert len(beats) >= 2


def test_audio_fades_present():
    sr = 1000
    clip1 = AudioClip(lambda t: [1.0], duration=1, fps=sr)
    clip2 = AudioClip(lambda t: [1.0], duration=1, fps=sr)
    clip1 = audio_fadeout(clip1, 0.2)
    clip2 = audio_fadein(clip2, 0.2)
    arr1 = clip1.to_soundarray(fps=sr)
    arr2 = clip2.to_soundarray(fps=sr)
    assert arr1[-1, 0] < 1.0 and arr2[0, 0] < 1.0


def test_final_audio_fade_rms():
    sr = 1000
    audio = AudioClip(lambda t: [1.0], duration=1, fps=sr)
    audio = audio_fadein(audio, 0.15)
    audio = audio_fadeout(audio, 0.15)
    arr = audio.to_soundarray(fps=sr)
    rms_start = np.sqrt(np.mean(arr[:150, 0] ** 2))
    rms_mid = np.sqrt(np.mean(arr[400:600, 0] ** 2))
    rms_end = np.sqrt(np.mean(arr[-150:, 0] ** 2))
    assert rms_start < rms_mid and rms_end < rms_mid


def test_audio_fit_trim_silence_loop(tmp_path):
    sr = 22050
    t = np.linspace(0, 0.4, int(sr * 0.4), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, y, sr)

    from ken_burns_reel.builder import _fit_audio_clip

    duration = 1.0
    a_trim = _fit_audio_clip(str(path), duration, "trim")
    assert a_trim.duration == pytest.approx(duration, 1e-3)

    a_sil = _fit_audio_clip(str(path), duration, "silence")
    arr = a_sil.to_soundarray(fps=sr)
    rms_tail = np.sqrt(np.mean(arr[-int(0.1 * sr) :, 0] ** 2))
    assert rms_tail < 0.05

    a_loop = _fit_audio_clip(str(path), duration, "loop")
    assert a_loop.duration == pytest.approx(duration, 1e-3)
