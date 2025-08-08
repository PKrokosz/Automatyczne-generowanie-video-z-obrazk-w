"""Utility helpers for video creation."""
from __future__ import annotations

from typing import Tuple

from PIL import Image
from moviepy.editor import CompositeVideoClip, ImageClip, TextClip


def overlay_caption(clip: ImageClip, text: str, size: Tuple[int, int]) -> ImageClip:
    """Overlay text caption onto a clip."""
    if not text:
        return clip
    W, H = size
    txt = (
        TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(W - 100, None),
        )
        .set_position(("center", H - 200))
        .set_duration(clip.duration)
        .fadein(0.5)
        .fadeout(0.5)
    )
    return CompositeVideoClip([clip, txt], size=size)


def smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop image to target aspect ratio while keeping center."""
    w, h = img.size
    src_ratio = w / h
    target_ratio = target_w / target_h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))
