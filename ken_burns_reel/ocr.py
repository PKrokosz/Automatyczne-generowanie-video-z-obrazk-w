"""OCR helpers."""
from __future__ import annotations

import os
from typing import Tuple

from PIL import Image
import pytesseract
from shutil import which


def extract_caption(img_path: str) -> str:
    """Extract text from an image using pytesseract."""
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return text.strip()


def verify_tesseract_available() -> None:
    """Ensure tesseract binary is available or exit."""
    binary = pytesseract.pytesseract.tesseract_cmd
    if not os.path.isfile(binary) and not which(binary):
        raise EnvironmentError(
            f"Tesseract OCR binary not found at: {binary}. Install Tesseract or update configuration."
        )
