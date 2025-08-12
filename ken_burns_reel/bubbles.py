"""Speech bubble detection and overlay helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2


@dataclass
class Bubble:
    """Detected speech bubble."""

    mask: np.ndarray  # float32 mask in [0,1]
    bbox: Tuple[int, int, int, int]
    score: float
    features: Dict[str, float]


@dataclass
class BubbleSprite:
    """Renderable speech bubble sprite."""

    img: np.ndarray  # RGBA pre-multiplied
    bbox: Tuple[int, int, int, int]
    z: int = 10  # render above panels


def _component_roundness(cnt: np.ndarray, area: float) -> float:
    peri = cv2.arcLength(cnt, True)
    if peri == 0:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def detect_bubbles(
    page_img: np.ndarray,
    ocr_boxes: List[Tuple[int, int, int, int]],
    cfg: Dict[str, Any] | None = None,
) -> List[Bubble]:
    """Detect speech bubbles on a page image.

    Parameters
    ----------
    page_img:
        RGB page image as ``np.ndarray``.
    ocr_boxes:
        List of OCR word bounding boxes ``(x, y, w, h)``.
    cfg:
        Detection configuration. Supported keys: ``min_area``, ``roundness_min``,
        ``feather_px``.
    """

    if cfg is None:
        cfg = {}
    min_area = int(cfg.get("min_area", 200))
    roundness_min = float(cfg.get("roundness_min", 0.3))
    feather_px = int(cfg.get("feather_px", 2))

    gray = cv2.cvtColor(page_img, cv2.COLOR_RGB2GRAY)
    white = cv2.inRange(gray, 200, 255)
    num, labels = cv2.connectedComponents(white)
    H, W = gray.shape

    chosen: set[int] = set()
    bubbles: List[Bubble] = []
    for (x, y, w, h) in ocr_boxes:
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        lbl = int(labels[cy, cx])
        if lbl == 0 or lbl in chosen:
            continue
        mask = (labels == lbl).astype(np.uint8) * 255
        chosen.add(lbl)
        area = float(mask.sum() / 255.0)
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        roundness = _component_roundness(cnt, area)
        if roundness < roundness_min:
            continue
        x0, y0, bw, bh = cv2.boundingRect(cnt)
        m = mask[y0 : y0 + bh, x0 : x0 + bw]
        if feather_px > 0:
            m = cv2.dilate(m, np.ones((feather_px, feather_px), np.uint8), 1)
            m = cv2.GaussianBlur(m, (0, 0), feather_px / 2)
        mask_f = (m.astype(np.float32) / 255.0).clip(0.0, 1.0)
        features = {"area": area, "roundness": roundness}
        bubbles.append(Bubble(mask=mask_f, bbox=(x0, y0, bw, bh), score=roundness, features=features))
    return bubbles


def bubble_sprite_from_page(page_img: np.ndarray, bubble: Bubble) -> BubbleSprite:
    """Extract an RGBA sprite for *bubble* from *page_img*."""

    x, y, w, h = bubble.bbox
    crop = page_img[y : y + h, x : x + w]
    alpha = (bubble.mask * 255).astype(np.uint8)
    rgba = np.dstack([crop, alpha])
    # premultiply
    rgb = rgba[:, :, :3].astype(np.float32)
    a = bubble.mask[..., None]
    premult = (rgb * a).astype(np.uint8)
    out = np.dstack([premult, alpha])
    return BubbleSprite(img=out, bbox=bubble.bbox)


def overlay_bubble_lift(frame: np.ndarray, sprite: BubbleSprite, t: float, dur: float = 0.18) -> None:
    """Overlay *sprite* onto *frame* with a subtle lift animation."""

    x, y, w, h = sprite.bbox
    p = min(1.0, max(0.0, t / max(1e-6, dur)))
    ease = 0.5 - 0.5 * np.cos(np.pi * p)
    scale = 1.0 + 0.02 * ease
    shadow_strength = 0.35 * ease
    img = sprite.img
    if scale != 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    sx = x - (img.shape[1] - w) // 2
    sy = y - (img.shape[0] - h) // 2
    # shadow
    if shadow_strength > 0:
        shadow = img[:, :, 3]
        shadow = cv2.GaussianBlur(shadow, (0, 0), 14)
        shadow = (shadow.astype(np.float32) * shadow_strength).astype(np.uint8)
        sh_rgba = np.zeros_like(img)
        sh_rgba[:, :, 3] = shadow
        _paste_rgba(frame, sh_rgba, sx, sy + 6)
    _paste_rgba(frame, img, sx, sy)


def _paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)
    roi = dst[y0:y1, x0:x1]
    src_roi = src[sy0:sy1, sx0:sx1]
    alpha = src_roi[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (src_roi[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
