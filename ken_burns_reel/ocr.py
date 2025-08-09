"""OCR helpers."""
from __future__ import annotations

from typing import Tuple

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
