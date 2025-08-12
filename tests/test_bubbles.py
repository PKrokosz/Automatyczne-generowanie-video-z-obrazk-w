import math
from typing import List, Tuple

import numpy as np
import cv2
import pytest

from ken_burns_reel.bubbles import detect_bubbles, bubble_sprite_from_page
from ken_burns_reel.builder import _paste_rgba_clipped


def _make_synthetic_page() -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    img = np.zeros((200, 200, 3), dtype=np.uint8) + 40
    cv2.circle(img, (60, 60), 30, (255, 255, 255), -1)
    cv2.circle(img, (60, 60), 30, (0, 0, 0), 2)
    cv2.circle(img, (150, 140), 40, (255, 255, 255), -1)
    cv2.circle(img, (150, 140), 40, (0, 0, 0), 2)
    boxes = [(50, 50, 20, 20), (140, 130, 20, 20)]
    return img, boxes


def test_bubble_detection_basic():
    img, boxes = _make_synthetic_page()
    bubbles = detect_bubbles(img, boxes, {"min_area": 200, "roundness_min": 0.6, "feather_px": 0})
    assert len(bubbles) == 2
    for b in bubbles:
        assert b.features["roundness"] > 0.6
        assert b.features["area"] > 2000


def test_bubble_overlay_no_clipping():
    page = np.zeros((200, 200, 3), dtype=np.uint8) + 40
    cv2.circle(page, (60, 60), 40, (255, 255, 255), -1)
    cv2.circle(page, (60, 60), 40, (0, 0, 0), 2)
    ocr_boxes = [(55, 55, 20, 20)]
    bubbles = detect_bubbles(page, ocr_boxes, {"min_area": 200, "roundness_min": 0.6, "feather_px": 0})
    spr = bubble_sprite_from_page(page, bubbles[0])
    canvas = np.zeros((200, 200, 4), dtype=np.uint8)
    panel_crop = page[20:120, 50:150]
    panel_rgba = np.dstack([panel_crop, np.full(panel_crop.shape[:2], 255, dtype=np.uint8)])
    _paste_rgba_clipped(canvas, panel_rgba, 50, 20)
    _paste_rgba_clipped(canvas, spr.img, spr.bbox[0], spr.bbox[1])
    tail_px = canvas[60, 30, :3]
    assert tail_px[0] > 200


def test_bg_stability_with_bubbles():
    img, boxes = _make_synthetic_page()
    bubbles = detect_bubbles(img, boxes, {"min_area": 200, "roundness_min": 0.6, "feather_px": 0})
    spr = bubble_sprite_from_page(img, bubbles[0])
    base = np.zeros((200, 200, 4), dtype=np.uint8)
    frame = base.copy()
    _paste_rgba_clipped(frame, spr.img, spr.bbox[0], spr.bbox[1])
    h, w = spr.img.shape[:2]
    mask = np.zeros((200, 200), dtype=bool)
    mask[spr.bbox[1]:spr.bbox[1]+h, spr.bbox[0]:spr.bbox[0]+w] = spr.img[:, :, 3] > 0
    diff = frame[:, :, :3].astype(np.int16) - base[:, :, :3].astype(np.int16)
    diff[mask] = 0
    rms = math.sqrt(np.mean(diff**2))
    assert rms == 0
