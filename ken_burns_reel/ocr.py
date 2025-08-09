"""OCR helpers."""
from __future__ import annotations

from typing import Dict, Tuple

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


def text_boxes_stats(img: Image.Image) -> Dict[str, float]:
    """Return basic stats about OCR-detected word boxes in *img*."""
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except Exception:
        return {"median_word_height": 0.0, "word_count": 0}
    heights = [int(h) for h, txt in zip(data.get("height", []), data.get("text", [])) if txt.strip()]
    median_h = float(np.median(heights)) if heights else 0.0
    words = sum(1 for txt in data.get("text", []) if txt.strip())
    return {"median_word_height": median_h, "word_count": words}
