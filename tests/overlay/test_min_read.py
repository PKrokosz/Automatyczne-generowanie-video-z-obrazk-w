import logging
import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from ken_burns_reel.panels import export_panels
from ken_burns_reel.builder import make_panels_overlay_sequence


def _make_page(tmp_path, cv2):
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(arr, (10, 10), (90, 90), (0, 0, 0), -1)
    cv2.rectangle(arr, (110, 10), (190, 90), (0, 0, 0), -1)
    p = tmp_path / "page.png"
    Image.fromarray(arr).save(p)
    return p


def test_overlay_warns_and_clamps_min_read(tmp_path, caplog):
    page = _make_page(tmp_path, cv2)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    caplog.set_level(logging.WARNING)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(300, 450),
        dwell=1.0,
        dwell_scale=0.5,
        dwell_mode="each",
        readability_ms=1400,
    )
    assert "min_read" in caplog.text
    assert clip.duration >= 1.4
