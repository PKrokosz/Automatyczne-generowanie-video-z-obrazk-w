"""Configuration helpers for ken_burns_reel."""
from __future__ import annotations

import os
from shutil import which

import pytesseract
from moviepy.config import change_settings

# ImageMagick binary can be provided via environment
IMAGEMAGICK_BINARY = os.environ.get("IMAGEMAGICK_BINARY")
if IMAGEMAGICK_BINARY:
    change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

# Ensure pytesseract uses system tesseract if available
pytesseract.pytesseract.tesseract_cmd = (
    which("tesseract") or pytesseract.pytesseract.tesseract_cmd
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
