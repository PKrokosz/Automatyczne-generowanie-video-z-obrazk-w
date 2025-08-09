"""Video transitions."""
from __future__ import annotations

from typing import Tuple

try:
    from moviepy.editor import CompositeVideoClip
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy import CompositeVideoClip


def slide_transition(prev_clip, next_clip, duration: float, size: Tuple[int, int], fps: int):
    """Simple horizontal slide transition between clips."""
    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)

    def move_left(t):
        return (-t / duration * W, 0)

    def move_right(t):
        return ((1 - t / duration) * W, 0)

    return (
        CompositeVideoClip(
            [tail.set_pos(move_left), head.set_pos(move_right)], size=size
        )
        .set_duration(duration)
        .set_fps(fps)
    )
