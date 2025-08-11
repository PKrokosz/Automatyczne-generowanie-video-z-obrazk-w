import numpy as np
from ken_burns_reel.color import srgb_to_linear16, linear16_to_srgb


def _hist_equal(a: np.ndarray, b: np.ndarray) -> bool:
    for c in range(a.shape[2]):
        h1, _ = np.histogram(a[..., c], bins=65536, range=(0, 65535))
        h2, _ = np.histogram(b[..., c], bins=65536, range=(0, 65535))
        if not np.array_equal(h1, h2):
            return False
    return True


def test_color_roundtrip_histogram():
    rng = np.random.default_rng(1234)
    img = rng.integers(0, 65536, size=(8, 8, 3), dtype=np.uint16)
    lin = srgb_to_linear16(img)
    out = linear16_to_srgb(lin)
    assert _hist_equal(img, out)
