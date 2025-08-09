"""Configuration helpers for ken_burns_reel."""
from __future__ import annotations

import os
from shutil import which

import pytesseract
try:
    from moviepy.config import change_settings
except ImportError:  # moviepy >=2.0
    change_settings = None

# Default binaries (can be overridden via environment)
IMAGEMAGICK_BINARY = os.environ.get(
    "IMAGEMAGICK_BINARY"
) or r"C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe"
if change_settings:
    change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

pytesseract.pytesseract.tesseract_cmd = (
    os.environ.get("TESSERACT_BINARY")
    or which("tesseract")
    or pytesseract.pytesseract.tesseract_cmd
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
