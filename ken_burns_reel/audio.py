"""Audio processing helpers."""
from __future__ import annotations

from typing import List

import librosa


def extract_beats(audio_path: str) -> List[float]:
    """Return beat times for an audio file."""
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    if not beat_times:
        beat_times = [0.0]
    if len(beat_times) < 2:
        beat_times.append(len(y) / sr)
    if beat_times[0] > 0.0:
        beat_times.insert(0, 0.0)
    return beat_times
