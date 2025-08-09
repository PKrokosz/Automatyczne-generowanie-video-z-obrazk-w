"""Helpers to resolve external binary paths.

This module centralizes discovery of ImageMagick and Tesseract executables
without requiring hard coded paths. Resolution order follows CLI arguments,
environment variables, existing library configuration and finally a search on
``PATH``.
"""
from __future__ import annotations

import os
import shutil
from typing import Optional

import moviepy.config as mpyconf
import pytesseract


def _validate_binary(path: str | None) -> Optional[str]:
    """Return *path* if it points to an existing executable."""
    if not path:
        return None
    if os.path.isfile(path) or shutil.which(path):
        return path
    return None


def resolve_imagemagick(cli_path: str | None = None) -> Optional[str]:
    """Resolve path to ImageMagick ``magick`` executable.

    Resolution order:
    1. explicit ``cli_path`` argument (e.g. ``--magick``)
    2. ``IMAGEMAGICK_BINARY`` environment variable
    3. MoviePy configuration
    4. ``magick`` discovered on ``PATH``
    The returned path is validated and applied to MoviePy configuration and
    ``os.environ``. Returns ``None`` if no candidate is found.
    """
    candidates = [
        cli_path,
        os.environ.get("IMAGEMAGICK_BINARY"),
        getattr(mpyconf, "IMAGEMAGICK_BINARY", None),
        shutil.which("magick"),
    ]
    for cand in candidates:
        path = _validate_binary(cand)
        if path:
            os.environ["IMAGEMAGICK_BINARY"] = path
            try:
                mpyconf.change_settings({"IMAGEMAGICK_BINARY": path})
            except Exception:
                pass
            return path
    return None


def resolve_tesseract(cli_path: str | None = None) -> Optional[str]:
    """Resolve path to the Tesseract binary.

    Resolution order mirrors :func:`resolve_imagemagick` but for Tesseract.
    The discovered path is stored in ``pytesseract.pytesseract.tesseract_cmd``
    and ``TESSERACT_BINARY`` environment variable. Returns ``None`` if the
    executable cannot be located.
    """
    candidates = [
        cli_path,
        os.environ.get("TESSERACT_BINARY"),
        getattr(pytesseract.pytesseract, "tesseract_cmd", None),
        shutil.which("tesseract"),
    ]
    for cand in candidates:
        path = _validate_binary(cand)
        if path:
            os.environ["TESSERACT_BINARY"] = path
            pytesseract.pytesseract.tesseract_cmd = path
            return path
    return None
