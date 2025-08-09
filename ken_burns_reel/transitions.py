"""Video transitions."""
from __future__ import annotations

from typing import Tuple
import math

import numpy as np
import cv2
from .utils import gaussian_blur, _set_fps


def ease_in_out(t: float) -> float:
    """Cosine ease-in-out for ``t`` in [0,1]."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_in(t: float) -> float:
    """Cosine ease-in for ``t`` in [0,1]."""
    return 1 - math.cos(0.5 * math.pi * t)


def ease_out(t: float) -> float:
    """Cosine ease-out for ``t`` in [0,1]."""
    return math.sin(0.5 * math.pi * t)


def _get_ease_fn(name: str):
    return {
        "linear": lambda t: t,
        "in": ease_in,
        "out": ease_out,
        "inout": ease_in_out,
    }.get(name, ease_in_out)

try:
    from moviepy.editor import CompositeVideoClip, VideoClip
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy import CompositeVideoClip, VideoClip


def slide_transition(prev_clip, next_clip, duration: float, size: Tuple[int, int], fps: int):
    """Simple horizontal slide transition between clips."""
    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)

    def move_left(t):
        return (-t / duration * W, 0)

    def move_right(t):
        return ((1 - t / duration) * W, 0)

    return _set_fps(
        CompositeVideoClip(
            [tail.set_pos(move_left), head.set_pos(move_right)], size=size
        ).set_duration(duration),
        fps,
    )


def smear_transition(
    prev_clip,
    next_clip,
    duration: float,
    size: Tuple[int, int],
    vec: Tuple[float, float],
    steps: int = 12,
    strength: float = 1.0,
    fps: int = 30,
):
    """Directional smear (pseudo motion blur) between clips."""

    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)
    dx, dy = vec

    def make_frame(t):
        p = t / duration
        frame_prev = tail.get_frame(t).astype(np.uint8)
        frame_next = head.get_frame(t).astype(np.uint8)
        acc_prev = np.zeros_like(frame_prev, dtype=np.float32)
        acc_next = np.zeros_like(frame_next, dtype=np.float32)
        for i in range(steps):
            s = i / max(1, steps - 1)
            M1 = np.float32([[1, 0, -dx * p * s * strength], [0, 1, -dy * p * s * strength]])
            warped1 = cv2.warpAffine(
                frame_prev, M1, (W, H), borderMode=cv2.BORDER_REFLECT
            )
            acc_prev += warped1.astype(np.float32)
            M2 = np.float32([[1, 0, (1 - p) * dx * (1 - s) * strength], [0, 1, (1 - p) * dy * (1 - s) * strength]])
            warped2 = cv2.warpAffine(
                frame_next, M2, (W, H), borderMode=cv2.BORDER_REFLECT
            )
            acc_next += warped2.astype(np.float32)
        smear_prev = acc_prev / steps
        smear_next = acc_next / steps
        alpha = p
        frame = (1 - alpha) * smear_prev + alpha * smear_next
        return np.clip(frame, 0, 255).astype(np.uint8)

    return _set_fps(VideoClip(make_frame, duration=duration), fps)


def whip_pan_transition(
    prev_clip,
    next_clip,
    duration: float,
    size: Tuple[int, int],
    vec: Tuple[float, float],
    fps: int = 30,
    ease: str = "inout",
):
    """Whip-pan style transition with easing and brightness dip."""

    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)
    dx, dy = vec
    ease_fn = _get_ease_fn(ease)

    def make_frame(t):
        p = ease_fn(t / duration)
        frame_prev = tail.get_frame(t).astype(np.uint8)
        frame_next = head.get_frame(t).astype(np.uint8)
        shift_prev = (-dx * p, -dy * p)
        shift_next = ((1 - p) * dx, (1 - p) * dy)
        M1 = np.float32([[1, 0, shift_prev[0]], [0, 1, shift_prev[1]]])
        M2 = np.float32([[1, 0, shift_next[0]], [0, 1, shift_next[1]]])
        prev_w = cv2.warpAffine(
            frame_prev, M1, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        next_w = cv2.warpAffine(
            frame_next, M2, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        blur = int(1 + 6 * p * (1 - p))
        if blur % 2 == 0:
            blur += 1
        prev_b = gaussian_blur(prev_w, (blur, blur))
        next_b = gaussian_blur(next_w, (blur, blur))
        alpha = p
        frame = (1 - alpha) * prev_b + alpha * next_b
        if 0.4 <= p <= 0.6:
            frame *= 0.9
        return np.clip(frame, 0, 255).astype(np.uint8)

    return _set_fps(VideoClip(make_frame, duration=duration), fps)


def smear_bg_crossfade_fg(
    tail_bg,
    head_bg,
    tail_fg,
    head_fg,
    duration: float,
    size: Tuple[int, int],
    vec: Tuple[float, float],
    steps: int = 12,
    strength: float = 1.0,
    fps: int = 30,
    bg_brightness_dip: float = 0.0,
    steps_auto: bool = False,
):
    """Smear transition for backgrounds with foreground crossfade."""

    W, H = size
    if steps_auto:
        steps = max(8, int(min(W, H) / 160))
    dip = max(0.0, min(0.15, bg_brightness_dip))
    tbg = tail_bg.subclip(tail_bg.duration - duration, tail_bg.duration)
    hbg = head_bg.subclip(0, duration)
    tfg = tail_fg.subclip(tail_fg.duration - duration, tail_fg.duration)
    hfg = head_fg.subclip(0, duration)
    dx, dy = vec

    def make_bg(t):
        p = t / duration
        frame_prev = tbg.get_frame(t).astype(np.uint8)
        frame_next = hbg.get_frame(t).astype(np.uint8)
        acc_prev = np.zeros_like(frame_prev, dtype=np.float32)
        acc_next = np.zeros_like(frame_next, dtype=np.float32)
        for i in range(steps):
            s = i / max(1, steps - 1)
            M1 = np.float32([[1, 0, -dx * p * s * strength], [0, 1, -dy * p * s * strength]])
            warped1 = cv2.warpAffine(
                frame_prev, M1, (W, H), borderMode=cv2.BORDER_REFLECT
            )
            acc_prev += warped1.astype(np.float32)
            M2 = np.float32(
                [[1, 0, (1 - p) * dx * (1 - s) * strength], [0, 1, (1 - p) * dy * (1 - s) * strength]]
            )
            warped2 = cv2.warpAffine(
                frame_next, M2, (W, H), borderMode=cv2.BORDER_REFLECT
            )
            acc_next += warped2.astype(np.float32)
        smear_prev = acc_prev / steps
        smear_next = acc_next / steps
        alpha = p
        frame = (1 - alpha) * smear_prev + alpha * smear_next
        if dip > 0 and 0.4 <= p <= 0.6:
            frame *= 1 - dip
        return np.clip(frame, 0, 255).astype(np.uint8)

    bg_clip = _set_fps(VideoClip(make_bg, duration=duration), fps)
    fg_clip = _set_fps(
        CompositeVideoClip(
            [tfg.crossfadeout(duration), hfg.crossfadein(duration)], size=size
        ).set_duration(duration),
        fps,
    )
    return _set_fps(
        CompositeVideoClip([bg_clip, fg_clip], size=size).set_duration(duration),
        fps,
    )
