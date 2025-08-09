import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
import soundfile as sf

from ken_burns_reel.bin_config import resolve_imagemagick, resolve_tesseract
from ken_burns_reel.builder import make_filmstrip, make_panels_cam_sequence

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
