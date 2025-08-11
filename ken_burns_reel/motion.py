from __future__ import annotations

"""Motion utilities for camera paths and subtle deterministic drift."""

from dataclasses import dataclass
import math
import random
from typing import Tuple


def arc_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    p: float,
    strength: float = 0.25,
) -> Tuple[float, float]:
    """Interpolate between ``start`` and ``end`` along an arc.

    Parameters
    ----------
    start, end:
        Start and end coordinates.
    p:
        Progress in ``[0, 1]``.
    strength:
        Relative strength of the perpendicular offset.
    """
    x = start[0] + (end[0] - start[0]) * p
    y = start[1] + (end[1] - start[1]) * p
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist > 1e-6:
        px, py = -dy / dist, dx / dist
        off = math.sin(math.pi * p) * dist * strength
        x += px * off
        y += py * off
    return x, y


@dataclass
class DriftParams:
    zoom_max: float
    xy_max: float
    rot_max: float


_DRIFT_CFG = {
    "bg": DriftParams(zoom_max=0.03, xy_max=0.006, rot_max=0.25),
    "fg": DriftParams(zoom_max=0.02, xy_max=0.004, rot_max=0.2),
}


def subtle_drift(kind: str, seed: int, p: float) -> Tuple[float, float, float, float]:
    """Return deterministic subtle drift parameters.

    ``p`` is progress ``[0, 1]``. ``seed`` ensures deterministic output.
    Returns ``(zoom, dx, dy, rot)`` where translation is expressed as a
    fraction of width/height and rotation in degrees.
    """
    cfg = _DRIFT_CFG[kind]
    rng = random.Random((seed << 1) ^ (0 if kind == "bg" else 1))
    zoom = 1.0 + rng.uniform(0.0, cfg.zoom_max) * p
    dx = rng.uniform(-cfg.xy_max, cfg.xy_max) * p
    dy = rng.uniform(-cfg.xy_max, cfg.xy_max) * p
    rot = rng.uniform(-cfg.rot_max, cfg.rot_max) * p
    return zoom, dx, dy, rot

def apply_transform(img, zoom: float, dx: float, dy: float, rot: float):
    """Apply zoom/translation/rotation drift to ``img``.

    Parameters are identical to :func:`subtle_drift` output. ``dx`` and ``dy``
    are expressed as fractions of width/height.
    """
    import cv2

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, zoom)
    M[0, 2] += dx * w
    M[1, 2] += dy * h
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
