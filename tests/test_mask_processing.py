import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from ken_burns_reel.panels import export_panels


def _make_page_with_hole(path: Path) -> None:
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], fill=(128, 128, 128))
    draw.rectangle([40, 40, 160, 160], fill="white")
    img.save(path)


def test_rect_fallback(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    _make_page_with_hole(page)
    out_dir = tmp_path / "panels" / "page_0001"
    out_dir.mkdir(parents=True)
    export_panels(
        str(page),
        str(out_dir),
        mode="mask",
        bleed=0,
        tight_border=0,
        feather=0,
        mask_fill_holes=0,
        mask_close=0,
        mask_rect_fallback=0.1,
    )
    panel_path = next(out_dir.glob("panel_*.png"))
    alpha = np.array(Image.open(panel_path).convert("RGBA"))[..., 3]
    assert alpha.min() == 255  # fallback to rect crop
