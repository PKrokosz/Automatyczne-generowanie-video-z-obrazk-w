"""Video transitions."""
from __future__ import annotations

from typing import Tuple
import math

import numpy as np
import cv2
from .utils import gaussian_blur, _set_fps
from .layers import page_shadow


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
        d = max(1e-6, duration)
        return (-t / d * W, 0)

    def move_right(t):
        d = max(1e-6, duration)
        return ((1 - t / d) * W, 0)

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
        p = t / max(1e-6, duration)
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
        p = ease_fn(t / max(1e-6, duration))
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


def fg_fade(panel_clip, duration: float, ease: str = "inout", fg_offset: float = 0.0):
    """Fade only the foreground alpha of ``panel_clip``.

    The background remains unchanged; only the mask (alpha) channel of the
    panel clip is attenuated. ``fg_offset`` allows shifting the fade in time to
    honour foreground offsets during transitions.
    """

    ease_fn = _get_ease_fn(ease)
    if panel_clip.mask is None:
        panel_clip = panel_clip.add_mask()
    mask = panel_clip.mask

    def mask_frame(t):
        p = ease_fn((t + fg_offset) / max(1e-6, duration))
        return mask.get_frame(t) * (1 - p)

    if hasattr(mask, "with_updated_frame_function"):
        faded_mask = mask.with_updated_frame_function(mask_frame)
    else:  # moviepy <2
        faded_mask = mask.fl(lambda gf, t: mask_frame(t))
    if hasattr(panel_clip, "with_mask"):
        return panel_clip.with_mask(faded_mask)
    return panel_clip.set_mask(faded_mask)


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
    bg_offset: float = 0.0,
    fg_offset: float = 0.0,
):
    """Smear transition for backgrounds with foreground crossfade."""

    W, H = size
    if steps_auto:
        steps = max(8, int(min(W, H) / 160))
    dip = max(0.0, min(0.15, bg_brightness_dip))
    tbg = tail_bg.subclip(
        max(0, tail_bg.duration - duration - bg_offset), tail_bg.duration - bg_offset
    )
    hbg = head_bg.subclip(bg_offset, bg_offset + duration)
    tfg = tail_fg.subclip(
        max(0, tail_fg.duration - duration - fg_offset), tail_fg.duration - fg_offset
    )
    hfg = head_fg.subclip(fg_offset, fg_offset + duration)
    dx, dy = vec

    def make_bg(t):
        p = t / max(1e-6, duration)
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


def overlay_lift(
    panel: np.ndarray,
    duration: float,
    lift: dict | None = None,
    fps: int = 30,
):
    """Lift-in effect for overlay panels.

    ``panel`` is expected in RGBA with straight alpha. The animation applies a
    subtle scale pop, shadow growth and alpha fade-in while keeping the
    background untouched.
    """

    if lift is None:
        lift = {"shadow": "grow", "scale": "pop", "alpha": "fade"}

    H, W = panel.shape[:2]

    def paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
        h, w = src.shape[:2]
        dst_h, dst_w = dst.shape[:2]
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(dst_w, x + w), min(dst_h, y + h)
        sx0, sy0 = max(0, -x), max(0, -y)
        sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        dst[dy0:dy1, dx0:dx1] = src[sy0:sy1, sx0:sx1]

    def make_rgba(t: float) -> np.ndarray:
        p = min(1.0, max(0.0, t / max(1e-6, duration)))
        scale = 0.9 + 0.1 * ease_out(p) if lift.get("scale") else 1.0
        alpha_f = ease_out(p) if lift.get("alpha") else 1.0
        shadow_s = ease_out(p) if lift.get("shadow") else 0.0
        arr = panel
        if scale != 1.0:
            nw = max(1, int(round(W * scale)))
            nh = max(1, int(round(H * scale)))
            arr = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_CUBIC)
        rgba = page_shadow(arr, strength=0.4 * shadow_s, blur=6, offset_xy=(6, 6))
        rgba = rgba.astype(np.float32)
        rgba[..., :3] *= alpha_f
        rgba[..., 3] *= alpha_f
        canvas = np.zeros((H, W, 4), dtype=np.float32)
        y0 = (H - rgba.shape[0]) // 2
        x0 = (W - rgba.shape[1]) // 2
        paste(canvas, rgba, x0, y0)
        return np.clip(canvas, 0, 255).astype(np.uint8)

    def make_frame(t: float) -> np.ndarray:
        return make_rgba(t)[:, :, :3]

    def make_mask(t: float) -> np.ndarray:
        return make_rgba(t)[:, :, 3] / 255.0

    clip = VideoClip(make_frame, duration=duration)
    try:
        mask = VideoClip(make_mask, is_mask=True, duration=duration)
        clip = clip.with_mask(mask)
    except TypeError:
        mask = VideoClip(make_mask, ismask=True, duration=duration)
        clip = clip.set_mask(mask)
    return _set_fps(clip, fps)
