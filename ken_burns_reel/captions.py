"""Caption rendering utilities."""
from __future__ import annotations

from typing import Tuple, Optional
import re

from moviepy.editor import CompositeVideoClip, TextClip

CAPTION_MAXLEN = 120
CAPTION_MIN_ALNUM = 3


def sanitize_caption(text: str) -> str:
    text = re.sub(r"[\\]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:CAPTION_MAXLEN]


def is_caption_meaningful(text: str) -> bool:
    return sum(ch.isalnum() for ch in (text or "")) >= CAPTION_MIN_ALNUM


def render_caption_clip(
    text: str, size: Tuple[int, int], margin: int = 50, method: str = "label"
) -> Optional[TextClip]:
    """Create a TextClip for *text* or return ``None`` if rendering fails."""
    text = sanitize_caption(text)
    if not is_caption_meaningful(text):
        return None
    W, _ = size
    try:
        clip = TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method=method,
            size=(W - 2 * margin, None),
        )
    except Exception as e:
        if method != "label":
            return render_caption_clip(text, size, margin, method="label")
        print(f"⚠️ Unable to render caption: {e}")
        return None
    return clip


def overlay_caption(
    clip, text: str, size: Tuple[int, int], margin: int = 50, method: str = "label"
):
    """Overlay *text* caption onto *clip* if possible."""
    txt = render_caption_clip(text, size, margin, method)
    if txt is None:
        return clip
    W, H = size
    txt = txt.set_position(("center", H - 200)).set_duration(clip.duration).fadein(0.3).fadeout(0.3)
    return CompositeVideoClip([clip, txt], size=size)
