"""OCR helpers."""
from __future__ import annotations

import os
from typing import Tuple

from PIL import Image
import pytesseract
from shutil import which

from .utils import sanitize_caption


def extract_caption(img_path: str) -> str:
    """Extract text from an image using pytesseract."""
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return sanitize_caption(text)


def verify_tesseract_available() -> None:
    """Ensure tesseract binary is available or exit."""
    binary = os.environ.get("TESSERACT_BINARY") or pytesseract.pytesseract.tesseract_cmd
    pytesseract.pytesseract.tesseract_cmd = binary
    if not os.path.isfile(binary) and not which(binary):
        raise EnvironmentError(
            f"Tesseract OCR binary not found at: {binary}. Install Tesseract or update configuration."
        )
    print(f"✅ Tesseract OCR: {binary}")
