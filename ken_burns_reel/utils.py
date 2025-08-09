"""Utility helpers for video creation."""
from __future__ import annotations

from typing import TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from moviepy.editor import ImageClip


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
