"""Utility helpers for video creation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
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


def gaussian_blur(
    img: np.ndarray,
    ksize: Tuple[int, int] = (0, 0),
    sigma: float = 0,
) -> np.ndarray:
    """Apply Gaussian blur with validated kernel parameters.

    Parameters
    ----------
    img:
        Input image array.
    ksize:
        Kernel width and height. Each dimension will be forced odd and
        clamped to ``min(width, height)`` of *img*.
    sigma:
        Standard deviation for Gaussian kernel. When ``ksize`` is ``(0, 0)``,
        ``sigma`` must be ``> 0``.
    """

    h, w = img.shape[:2]
    limit = min(w, h)
    kw, kh = ksize
    if kw <= 0 or kh <= 0:
        if sigma <= 0:
            raise ValueError("sigma must be > 0 when kernel size is (0,0)")
        k = (0, 0)
    else:
        kw = min(kw, limit)
        kh = min(kh, limit)
        if kw % 2 == 0:
            kw = max(1, kw - 1)
        if kh % 2 == 0:
            kh = max(1, kh - 1)
        k = (kw, kh)
    return cv2.GaussianBlur(img, k, sigma)


def _set_fps(clip, fps):
    """Set frames-per-second on a clip for moviepy 1.x/2.x."""
    return clip.set_fps(fps) if hasattr(clip, "set_fps") else clip.with_fps(fps)
