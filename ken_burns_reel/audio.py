"""Audio processing helpers."""
from __future__ import annotations

from typing import List

import librosa


def extract_beats(audio_path: str) -> List[float]:
    """Return beat times for an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return beat_times.tolist()
