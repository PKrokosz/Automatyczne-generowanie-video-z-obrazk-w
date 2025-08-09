import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from ken_burns_reel.panels import export_panels


def _make_page_face_bubble(path: Path) -> None:
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 180, 180], fill=(128, 128, 128))
    draw.ellipse([40, 40, 80, 80], fill="white")  # speech bubble
    draw.ellipse([110, 80, 150, 120], fill="white")  # face
    img.save(path)


def test_mask_fill_holes_speech_and_skin(tmp_path: Path) -> None:
    page = tmp_path / "page.png"
    _make_page_face_bubble(page)
    out_dir = tmp_path / "panels" / "page_0001"
    out_dir.mkdir(parents=True)
    export_panels(
        str(page),
        str(out_dir),
        mode="mask",
        bleed=0,
        tight_border=0,
        feather=0,
        roughen=0.0,
        mask_rect_fallback=1.0,
    )
    panel_path = next(out_dir.glob("panel_*.png"))
    alpha = np.array(Image.open(panel_path).convert("RGBA"))[..., 3]
    assert alpha[60, 60] >= 250  # bubble interior
    assert alpha[100, 100] >= 250  # face interior
