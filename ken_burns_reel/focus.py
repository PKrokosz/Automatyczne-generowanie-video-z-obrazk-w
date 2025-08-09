"""Focus point detection."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2
from PIL import Image
import logging


def detect_focus_point(img: Image.Image) -> Tuple[int, int]:
    """Detect a point of interest in an image.

    If a face is detected the centre of the first face is used.
    Otherwise a brightness weighted centroid is returned.
    """
    gray = np.array(img.convert("L"))
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return (x + w // 2, y + h // 2)
    brightness = gray.astype(float)
    y_indices, x_indices = np.indices(brightness.shape)
    total = brightness.sum()
    eps = 1e-6
    if total <= eps:
        h, w = gray.shape
        logging.warning("detect_focus_point: black frame – using centre")
        return (w // 2, h // 2)
    den = max(eps, total)
    x = int((x_indices * brightness).sum() / den)
    y = int((y_indices * brightness).sum() / den)
    return (x, y)
