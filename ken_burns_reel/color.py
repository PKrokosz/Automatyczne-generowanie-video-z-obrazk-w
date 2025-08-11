"""Colour space utilities.

These helpers ensure a consistent sRGB → linear → sRGB path using
16‑bit precision, minimising cumulative rounding errors.  The conversion
formulae follow the sRGB specification and operate on NumPy arrays.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear16(img: np.ndarray) -> np.ndarray:
    """Convert 16‑bit sRGB values to linear light floats.

    Parameters
    ----------
    img:
        ``numpy`` array of dtype ``uint16``.
    Returns
    -------
    numpy.ndarray
        Float64 array in range ``[0,1]`` representing linear light values.
    """
    arr = img.astype(np.float64) / 65535.0
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def linear16_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Convert linear light floats back to 16‑bit sRGB values."""
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    srgb = np.clip(srgb, 0.0, 1.0)
    return (srgb * 65535.0 + 0.5).astype(np.uint16)
