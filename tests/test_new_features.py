import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import pytest
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.AudioClip import AudioClip

from ken_burns_reel.bin_config import resolve_imagemagick, resolve_tesseract
from ken_burns_reel.builder import (
    _export_profile,
    make_panels_cam_clip,
    make_panels_cam_sequence,
)
from ken_burns_reel.panels import detect_panels


IM = resolve_imagemagick()
TE = resolve_tesseract()


def _sample_image(path: Path) -> None:
    img = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 280, 280], outline="black", width=5)
    img.save(path)


@pytest.mark.skipif(IM is None or TE is None, reason="requires ImageMagick and Tesseract")
def test_underlay_brightness(tmp_path: Path) -> None:
    path = tmp_path / "page.png"
    _sample_image(path)
    clip = make_panels_cam_clip(
        str(path), target_size=(120, 120), bg_mode="blur", page_scale=0.9
    )
    frame = clip.get_frame(0)
    bg_patch = frame[0:20, 0:20]
    fg_patch = frame[40:80, 40:80]

    def lum(a: np.ndarray) -> float:
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    assert lum(bg_patch) <= lum(fg_patch) * 0.9


@pytest.mark.skipif(IM is None or TE is None, reason="requires ImageMagick and Tesseract")
def test_nested_suppression() -> None:
    img = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 190], outline="black", width=5)
    draw.rectangle([60, 60, 140, 140], outline="black", width=3)
    boxes = detect_panels(img)
    assert len(boxes) == 1


def test_preview_profile_smaller(tmp_path: Path) -> None:
    def make_frame(t):
        rng = np.random.default_rng(int(t * 30))
        return rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)

    clip = VideoClip(make_frame, duration=2).set_fps(30)
    audio = AudioClip(lambda t: [0.5], duration=2, fps=44100)
    clip = clip.set_audio(audio)
    prof_q = _export_profile("quality", "h264", (64, 64))
    prof_p = _export_profile("preview", "h264", (64, 64))
    out_q = tmp_path / "q.mp4"
    out_p = tmp_path / "p.mp4"
    clip.write_videofile(
        str(out_q),
        fps=prof_q["fps"],
        codec=prof_q["codec"],
        audio_codec=prof_q["audio_codec"],
        audio_bitrate=prof_q["audio_bitrate"],
        ffmpeg_params=prof_q["ffmpeg_params"],
        preset=prof_q["preset"],
    )
    clip.write_videofile(
        str(out_p),
        fps=prof_p["fps"],
        codec=prof_p["codec"],
        audio_codec=prof_p["audio_codec"],
        audio_bitrate=prof_p["audio_bitrate"],
        ffmpeg_params=prof_p["ffmpeg_params"],
        preset=prof_p["preset"],
    )
    assert out_p.stat().st_size <= out_q.stat().st_size * 0.6


@pytest.mark.skipif(IM is None or TE is None, reason="requires ImageMagick and Tesseract")
def test_custom_size(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    _sample_image(path)
    clip = make_panels_cam_sequence([str(path)], target_size=(1920, 1080), dwell=0.1, travel=0.1)
    assert clip.w == 1920 and clip.h == 1080
