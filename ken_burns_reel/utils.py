"""Utility helpers for video creation."""
from __future__ import annotations

from typing import Tuple

import re
from typing import TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from moviepy.editor import CompositeVideoClip, ImageClip, TextClip

# Caption configuration
CAPTION_MAXLEN = 120
CAPTION_MIN_ALNUM = 3


def sanitize_caption(text: str) -> str:
    text = re.sub(r"[\\]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:CAPTION_MAXLEN]


def is_caption_meaningful(text: str) -> bool:
    return sum(ch.isalnum() for ch in (text or "")) >= CAPTION_MIN_ALNUM


def overlay_caption(clip: "ImageClip", text: str, size: Tuple[int, int]) -> "ImageClip":
    """Overlay text caption onto a clip."""
    from moviepy.editor import CompositeVideoClip, ImageClip, TextClip
    text = sanitize_caption(text)
    if not is_caption_meaningful(text):
        return clip
    W, H = size
    try:
        txt = TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(W - 100, None),
        )
    except Exception as e:
        print(f"⚠️ TextClip fallback to 'label': {e}")
        txt = TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="label",
        )
    txt = txt.set_position(("center", H - 200)).set_duration(clip.duration).fadein(0.3).fadeout(0.3)
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
