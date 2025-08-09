"""Video transitions."""
from __future__ import annotations

from typing import Tuple
import math

import numpy as np
import cv2

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

    return (
        CompositeVideoClip(
            [tail.set_pos(move_left), head.set_pos(move_right)], size=size
        )
        .set_duration(duration)
        .set_fps(fps)
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

    return VideoClip(make_frame, duration=duration).set_fps(fps)


def whip_pan_transition(
    prev_clip,
    next_clip,
    duration: float,
    size: Tuple[int, int],
    vec: Tuple[float, float],
    fps: int = 30,
):
    """Whip-pan style transition with eased velocity and brightness dip."""

    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)
    dx, dy = vec

    def ease(t: float) -> float:
        a = t**3
        b = (1 - t) ** 3
        return a / (a + b) if a + b > 0 else t

    def make_frame(t):
        p = ease(t / duration)
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
        prev_b = cv2.GaussianBlur(prev_w, (blur, blur), 0)
        next_b = cv2.GaussianBlur(next_w, (blur, blur), 0)
        alpha = p
        frame = (1 - alpha) * prev_b + alpha * next_b
        if 0.4 <= p <= 0.6:
            frame *= 0.9
        return np.clip(frame, 0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration).set_fps(fps)


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
):
    """Smear transition applied only to backgrounds with foreground crossfade."""

    W, H = size
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
        return np.clip(frame, 0, 255).astype(np.uint8)

    bg_clip = VideoClip(make_bg, duration=duration).set_fps(fps)
    fg_clip = (
        CompositeVideoClip(
            [tfg.crossfadeout(duration), hfg.crossfadein(duration)], size=size
        )
        .set_duration(duration)
        .set_fps(fps)
    )
    return (
        CompositeVideoClip([bg_clip, fg_clip], size=size)
        .set_duration(duration)
        .set_fps(fps)
    )
