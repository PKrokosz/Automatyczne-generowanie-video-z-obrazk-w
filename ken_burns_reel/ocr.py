"""OCR helpers."""
from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
import pytesseract

from .bin_config import resolve_tesseract
from .captions import sanitize_caption


def extract_caption(img_path: str) -> str:
    """Extract text from an image using pytesseract."""
    with Image.open(img_path) as img:
        text = pytesseract.image_to_string(img)
    return sanitize_caption(text)


def verify_tesseract_available() -> None:
    """Ensure tesseract binary is available or raise an error."""
    binary = resolve_tesseract()
    if not binary:
        raise EnvironmentError(
            "Tesseract OCR binary not found. Install Tesseract or update configuration."
        )
    print(f"✅ Tesseract OCR: {binary}")


def page_ocr_data(img: Image.Image) -> Dict:
    """Return raw OCR data for an entire page."""
    try:
        return pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return {"text": []}


def text_boxes_stats(img: Image.Image) -> Dict[str, float]:
    """Return basic stats about OCR-detected word boxes in *img*."""
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return {"median_word_height": 0.0, "word_count": 0}
    h_img = img.size[1]
    heights: List[int] = []
    words = 0
    for h, txt in zip(data.get("height", []), data.get("text", [])):
        if not txt.strip():
            continue
        if h < 4 or h > 0.5 * h_img:
            continue
        heights.append(int(h))
        words += 1
    median_h = float(np.median(heights)) * 0.7 if heights else 0.0
    return {"median_word_height": median_h, "word_count": words}
