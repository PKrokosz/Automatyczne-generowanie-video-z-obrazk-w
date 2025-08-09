import numpy as np
import soundfile as sf
import librosa

from ken_burns_reel.audio import extract_beats
from moviepy.audio.AudioClip import AudioClip
from moviepy.audio.fx.audio_fadein import audio_fadein
from moviepy.audio.fx.audio_fadeout import audio_fadeout


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
