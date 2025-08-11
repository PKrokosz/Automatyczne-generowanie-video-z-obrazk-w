import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from ken_burns_reel.builder import make_panels_overlay_sequence
from ken_burns_reel.panels import export_panels, detect_panels, order_panels_lr_tb


def _make_two_panel_page(path: Path) -> None:
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 90, 90], fill=(50, 50, 50))
    draw.rectangle([110, 10, 190, 90], fill=(80, 80, 80))
    img.save(path)


def test_bg_is_stable_vs_fg_motion(tmp_path: Path) -> None:
    page_path = tmp_path / "page.png"
    _make_two_panel_page(page_path)
    panels_dir = tmp_path / "panels" / "page_0001"
    panels_dir.mkdir(parents=True)
    export_panels(
        str(page_path),
        str(panels_dir),
        mode="rect",
        bleed=0,
        tight_border=0,
        feather=0,
        roughen=0.0,
    )
    clip = make_panels_overlay_sequence(
        [str(page_path)],
        str(tmp_path / "panels"),
        target_size=(200, 100),
        fps=10,
        dwell=0.2,
        travel=0.0,
        trans_dur=0.0,
        parallax_bg=0.05,
        parallax_fg=0.0,
        bg_blur=0.0,
    )
    frame1 = clip.get_frame(0.1)
    frame2 = clip.get_frame(0.25)
    lum1 = frame1.mean(axis=2)
    lum2 = frame2.mean(axis=2)
    diff = np.abs(lum1 - lum2)
    with Image.open(page_path) as im:
        boxes = order_panels_lr_tb(detect_panels(im))
        scale_x = 200 / im.width
        scale_y = 100 / im.height
    fg_mask = np.zeros(diff.shape, dtype=bool)
    for x, y, w, h in boxes:
        xs = int(round(x * scale_x))
        ys = int(round(y * scale_y))
        xe = int(round((x + w) * scale_x))
        ye = int(round((y + h) * scale_y))
        fg_mask[ys:ye, xs:xe] = True
    fg_delta = diff[fg_mask].mean()
    bg_delta = diff[~fg_mask].mean()
    assert bg_delta < 0.25 * fg_delta


def test_fg_fade_transition_background_static(tmp_path: Path) -> None:
    page_path = tmp_path / "page.png"
    _make_two_panel_page(page_path)
    panels_dir = tmp_path / "panels" / "page_0001"
    panels_dir.mkdir(parents=True)
    export_panels(
        str(page_path),
        str(panels_dir),
        mode="rect",
        bleed=0,
        tight_border=0,
        feather=0,
        roughen=0.0,
    )
    clip = make_panels_overlay_sequence(
        [str(page_path)],
        str(tmp_path / "panels"),
        target_size=(200, 100),
        fps=10,
        dwell=0.2,
        travel=0.0,
        settle=0.0,
        trans="fg-fade",
        trans_dur=0.1,
        parallax_bg=0.0,
        parallax_fg=0.0,
        bg_blur=0.0,
    )
    frame_before = clip.get_frame(0.15)
    frame_trans = clip.get_frame(0.25)
    diff = np.abs(frame_before.astype(float) - frame_trans.astype(float)).mean(axis=2)
    with Image.open(page_path) as im:
        boxes = order_panels_lr_tb(detect_panels(im))
        scale_x = 200 / im.width
        scale_y = 100 / im.height
    fg_mask = np.zeros(diff.shape, dtype=bool)
    for x, y, w, h in boxes:
        xs = int(round(x * scale_x))
        ys = int(round(y * scale_y))
        xe = int(round((x + w) * scale_x))
        ye = int(round((y + h) * scale_y))
        fg_mask[ys:ye, xs:xe] = True
    fg_delta = diff[fg_mask].mean()
    bg_delta = diff[~fg_mask].mean()
    assert bg_delta < 0.25 * fg_delta
