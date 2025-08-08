"""Build a Ken Burns style video synchronized to audio beats."""
from __future__ import annotations

import os
import sys
import re
from typing import List, Tuple

import numpy as np
from PIL import Image
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop
from shutil import which
import moviepy.config as mpyconf
import pytesseract

# --- BINARIES (Windows) ---
IMAGEMAGICK_BINARY = r"C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe"

os.environ["IMAGEMAGICK_BINARY"] = IMAGEMAGICK_BINARY
mpyconf.change_settings({"IMAGEMAGICK_BINARY": IMAGEMAGICK_BINARY})

pytesseract.pytesseract.tesseract_cmd = (
    which("tesseract") or r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)


def verify_tesseract_available() -> None:
    binary = pytesseract.pytesseract.tesseract_cmd
    if not os.path.isfile(binary) and not which(binary):
        raise EnvironmentError(f"Tesseract OCR not found at: {binary}")
    print(f"✅ Tesseract OCR: {binary}")


# --- MONTAGE CONFIG ---
FORMAT: Tuple[int, int] = (1080, 1920)
FPS = 30

BEATS_PER_IMAGE = 4
MIN_CLIP = 1.2
MAX_CLIP = 3.5
TRANSITION = 0.3

ZOOM_START = 1.14
ZOOM_END = 1.06
PAN_MAX = 0.18

CAPTION_MAXLEN = 120
CAPTION_MIN_ALNUM = 3


def sanitize_caption(text: str) -> str:
    text = re.sub(r"[\\]+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:CAPTION_MAXLEN]


def is_caption_meaningful(text: str) -> bool:
    return sum(ch.isalnum() for ch in (text or "")) >= CAPTION_MIN_ALNUM


def extract_caption(img_path: str) -> str:
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return sanitize_caption(text)


def overlay_caption(clip: ImageClip, text: str, size: Tuple[int, int]) -> ImageClip:
    text = sanitize_caption(text)
    if not is_caption_meaningful(text):
        return clip
    W, H = size
    try:
        txt = TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(W - 100, None),
        )
    except Exception as e:
        print(f"⚠️ TextClip fallback to 'label': {e}")
        txt = TextClip(
            text,
            fontsize=50,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="label",
        )
    txt = txt.set_position(("center", H - 200)).set_duration(clip.duration).fadein(0.3).fadeout(0.3)
    return CompositeVideoClip([clip, txt], size=size)


def ken_burns_scroll(
    image_path: str,
    size: Tuple[int, int],
    duration: float,
    fps: int,
    time_span: Tuple[float, float],
    focus_point: Tuple[int, int] | None,
    caption: str,
) -> CompositeVideoClip:
    W, H = size
    img_clip = ImageClip(image_path).set_duration(duration)
    fx, fy = focus_point if focus_point else (img_clip.w // 2, img_clip.h // 2)

    start_zoom = ZOOM_START
    end_zoom = ZOOM_END

    def zoom(t: float) -> float:
        return start_zoom + (end_zoom - start_zoom) * (t / duration)

    def x_center(t: float) -> float:
        p = t / duration
        dx_raw = (img_clip.w // 2 - fx) * p
        dx = np.clip(dx_raw, -img_clip.w * PAN_MAX, img_clip.w * PAN_MAX)
        return img_clip.w // 2 - dx

    def y_center(t: float) -> float:
        p = t / duration
        dy_raw = (img_clip.h // 2 - fy) * p
        dy = np.clip(dy_raw, -img_clip.h * PAN_MAX, img_clip.h * PAN_MAX)
        return img_clip.h // 2 - dy

    zoomed = img_clip.resize(lambda t: zoom(t))
    scrolled = zoomed.crop(width=W, height=H, x_center=x_center, y_center=y_center)
    scrolled = scrolled.set_duration(duration).set_fps(fps)
    return overlay_caption(scrolled, caption, size)


def chunk_beats(beat_times: List[float], beats_per_image: int) -> List[Tuple[float, float]]:
    chunks: List[Tuple[float, float]] = []
    i = 0
    last = len(beat_times) - 1
    while i + beats_per_image <= last:
        t0 = float(beat_times[i])
        t1 = float(beat_times[i + beats_per_image])
        chunks.append((t0, t1))
        i += beats_per_image
    if not chunks:
        return [(3.0 * k, 3.0 * (k + 1)) for k in range(0, len(beat_times) - 1)]
    return chunks


def make_filmstrip(input_folder: str) -> str:
    from ken_burns_reel.audio import extract_beats
    from ken_burns_reel.focus import detect_focus_point
    from ken_burns_reel.transitions import slide_transition

    image_files = sorted(
        f
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    image_paths = [os.path.join(input_folder, f) for f in image_files]
    if not image_paths:
        raise FileNotFoundError("No images found in input folder")

    audio_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in {".mp3", ".wav", ".m4a"}
    ]
    if not audio_files:
        raise FileNotFoundError("No audio file found in input folder")
    audio_path = os.path.join(input_folder, audio_files[0])

    beat_times = extract_beats(audio_path)
    chunks = chunk_beats(beat_times, BEATS_PER_IMAGE)

    size = FORMAT
    fps = FPS
    images = image_paths
    n = min(len(images), len(chunks))
    clips: List[CompositeVideoClip] = []
    for i in range(n):
        path = images[i]
        t0, t1 = chunks[i]
        duration = max(MIN_CLIP, min(t1 - t0, MAX_CLIP))
        caption = extract_caption(path)
        focus_point = detect_focus_point(Image.open(path))
        clip = ken_burns_scroll(path, size, duration, fps, (t0, t1), focus_point, caption)
        clips.append(clip)
        if i < n - 1:
            next_path = images[i + 1]
            next_t0, next_t1 = chunks[i + 1]
            next_duration = max(MIN_CLIP, min(next_t1 - next_t0, MAX_CLIP))
            next_caption = extract_caption(next_path)
            next_focus = detect_focus_point(Image.open(next_path))
            next_clip = ken_burns_scroll(
                next_path, size, next_duration, fps, (next_t0, next_t1), next_focus, next_caption
            )
            clips.append(
                slide_transition(clip, next_clip, TRANSITION, size, fps)
            )
    final = concatenate_videoclips(clips, method="compose")
    final = final.set_audio(AudioFileClip(audio_path))
    output_path = os.path.join(input_folder, "final_video.mp4")
    final.write_videofile(output_path, fps=fps, codec="libx264")
    return output_path


if __name__ == "__main__":
    verify_tesseract_available()
    make_filmstrip(sys.argv[1])
