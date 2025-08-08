import numpy as np
import soundfile as sf
import librosa

from ken_burns_reel.audio import extract_beats


def test_extract_beats(tmp_path):
    sr = 22050
    y = librosa.clicks(times=[0.0, 1.0], sr=sr, length=sr * 2)
    path = tmp_path / "clicks.wav"
    sf.write(path, y, sr)
    beats = extract_beats(str(path))
    assert len(beats) >= 2
