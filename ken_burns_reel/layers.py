"""Layer effects utilities."""
from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import cv2
import hashlib


# simple in-memory cache for page effects
_SHADOW_CACHE: Dict[Tuple[str, float, int, Tuple[int, int]], np.ndarray] = {}


def _premultiply(arr: np.ndarray) -> np.ndarray:
    """Return *arr* with RGB channels pre-multiplied by alpha."""
    if arr.shape[-1] == 4:
        rgb = arr[..., :3].astype(np.float32)
        alpha = arr[..., 3:4].astype(np.float32) / 255.0
        rgb = (rgb * alpha).astype(np.uint8)
        out = arr.copy()
        out[..., :3] = rgb
        return out
    return arr


def _paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Paste *src* onto *dst* at ``(x, y)`` without clipping checks."""
    h, w = src.shape[:2]
    dst[y : y + h, x : x + w] = src


def page_shadow(
    img: np.ndarray,
    strength: float = 0.25,
    blur: int = 8,
    offset_xy: Tuple[int, int] = (6, 6),
) -> np.ndarray:
    """Apply drop shadow to ``img`` and return RGBA with premultiplied alpha.

    The result is cached based on image content and parameters.
    """

    if img.shape[-1] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        src = np.dstack([img, alpha])
    else:
        src = img.copy()
    h, w = src.shape[:2]
    key_hash = hashlib.sha1(src.tobytes()).hexdigest()
    key = (key_hash, float(strength), int(blur), tuple(offset_xy))
    if key in _SHADOW_CACHE:
        return _SHADOW_CACHE[key].copy()

    ox, oy = offset_xy
    pad_l = max(blur - min(0, ox), 0)
    pad_t = max(blur - min(0, oy), 0)
    pad_r = max(blur + max(0, ox), 0)
    pad_b = max(blur + max(0, oy), 0)
    canvas = np.zeros((h + pad_t + pad_b, w + pad_l + pad_r, 4), dtype=np.uint8)

    alpha = src[:, :, 3]
    shadow = cv2.GaussianBlur(alpha, (0, 0), blur)
    shadow = (shadow.astype(np.float32) * strength).clip(0, 255).astype(np.uint8)
    shadow_rgba = np.zeros_like(src)
    shadow_rgba[:, :, 3] = shadow

    _paste_rgba(canvas, shadow_rgba, pad_l + ox, pad_t + oy)
    _paste_rgba(canvas, _premultiply(src), pad_l, pad_t)

    _SHADOW_CACHE[key] = canvas
    return canvas.copy()
