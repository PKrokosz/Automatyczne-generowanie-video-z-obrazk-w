import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
import soundfile as sf
import pytesseract

from ken_burns_reel.bin_config import resolve_imagemagick, resolve_tesseract
from ken_burns_reel.builder import (
    make_filmstrip,
    make_panels_cam_sequence,
    make_panels_cam_clip,
    apply_clahe_rgb,
)
from ken_burns_reel.ocr import text_boxes_stats
from ken_burns_reel.panels import detect_panels, order_panels_lr_tb
from ken_burns_reel.builder import _fit_window_to_box

IM = resolve_imagemagick()
TE = resolve_tesseract()

pytestmark = pytest.mark.skipif(
    IM is None or TE is None, reason="requires ImageMagick and Tesseract"
)


def _create_image(path: Path) -> None:
    img = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 350, 350], outline="black", width=5)
    img.save(path)


def _create_audio(path: Path, duration: float = 2.0, sr: int = 22050) -> None:
    data = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, data, sr)


def test_builder_and_panels_sequence(tmp_path: Path) -> None:
    for i in range(2):
        _create_image(tmp_path / f"img{i}.png")
    _create_audio(tmp_path / "audio.wav")

    make_filmstrip(str(tmp_path))
    out = tmp_path / "final_video.mp4"
    assert out.exists() and out.stat().st_size > 0

    clip = make_panels_cam_sequence(
        [str(tmp_path / f"img{i}.png") for i in range(2)],
        target_size=(64, 64),
        dwell=0.1,
        travel=0.1,
    )
    out2 = tmp_path / "panels.mp4"
    clip.write_videofile(str(out2), fps=24, codec="libx264")
    assert out2.exists() and out2.stat().st_size > 0


def test_panels_easing_and_settle(tmp_path: Path) -> None:
    # create image with two colored panels
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], outline="black", width=5, fill="red")
    draw.rectangle([210, 10, 390, 190], outline="black", width=5, fill="green")
    img_path = tmp_path / "two.png"
    img.save(img_path)

    clip = make_panels_cam_clip(
        str(img_path),
        target_size=(64, 32),
        dwell=0.6,
        travel=0.4,
        settle=0.1,
        easing="ease",
    )
    assert clip.duration == pytest.approx(1.0, 1e-2)
    f1 = clip.get_frame(0.6 + 0.1)  # early travel
    f2 = clip.get_frame(0.6 + 0.2)
    f3 = clip.get_frame(0.6 + 0.3)
    r1 = f1[:, :, 1].mean()  # green channel increases as we move right
    r2 = f2[:, :, 1].mean()
    r3 = f3[:, :, 1].mean()
    assert (r2 - r1) > 0 and (r3 - r2) > (r2 - r1)


def test_ocr_readability_zoom(tmp_path: Path) -> None:
    img = Image.new("RGB", (500, 400), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 450, 350], outline="black", width=5)
    draw.text((60, 60), "tiny text", fill="black")
    path = tmp_path / "small_text.png"
    img.save(path)

    clip = make_panels_cam_clip(str(path), target_size=(64, 64), dwell=0.3, travel=0.2)
    assert clip.duration > 0

    with Image.open(path) as im:
        boxes = order_panels_lr_tb(detect_panels(im))
        arr = np.array(im)
        W, H = im.size
    x, y, w, h = boxes[0]
    _, _, ww, wh = _fit_window_to_box(W, H, boxes[0], (64, 64))
    stats = text_boxes_stats(Image.fromarray(arr[y : y + h, x : x + w]))
    med = stats.get("median_word_height", 0.0)
    zoomed_wh = wh
    if med and med / h < 0.035 and wh > h + 48:
        zoomed_wh = max(int(wh / 1.06), h + 48)
    assert zoomed_wh < wh


def test_adaptive_threshold_fallback() -> None:
    img = Image.new("RGB", (400, 200), (255, 255, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], fill="black")
    draw.rectangle([210, 10, 390, 190], fill="black")
    boxes = detect_panels(img)
    assert len(boxes) >= 2


def test_dwell_mode_first_vs_each(tmp_path: Path) -> None:
    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], outline="black", width=5)
    draw.rectangle([210, 10, 390, 190], outline="black", width=5)
    path = tmp_path / "dwell.png"
    img.save(path)

    clip_first = make_panels_cam_clip(
        str(path), target_size=(64, 32), dwell=0.5, travel=0.2, dwell_mode="first"
    )
    clip_each = make_panels_cam_clip(
        str(path), target_size=(64, 32), dwell=0.5, travel=0.2, dwell_mode="each"
    )
    assert clip_each.duration > clip_first.duration


def test_ocr_page_cache(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_image_to_data(*args, **kwargs):
        calls.append(1)
        return {"text": [], "left": [], "top": [], "width": [], "height": []}

    monkeypatch.setattr(pytesseract, "image_to_data", fake_image_to_data)
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 90, 90], outline="black")
    draw.rectangle([110, 10, 190, 90], outline="black")
    path = tmp_path / "ocr.png"
    img.save(path)
    make_panels_cam_clip(str(path), target_size=(64, 32))
    assert len(calls) == 1


def test_clahe_gamma_clamp() -> None:
    arr = np.full((50, 50, 3), 5, dtype=np.uint8)
    out = apply_clahe_rgb(arr)
    assert out.mean() < 100
