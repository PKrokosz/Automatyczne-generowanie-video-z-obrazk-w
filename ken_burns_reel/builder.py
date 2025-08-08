"""Core building logic for the Ken Burns reel."""
from __future__ import annotations

import glob
import os
from typing import List, Tuple

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop
from PIL import Image

from .audio import extract_beats
from .focus import detect_focus_point
from .ocr import extract_caption
from .utils import overlay_caption
from .config import IMAGE_EXTS, AUDIO_EXTS


ScreenSize = Tuple[int, int]


def ken_burns_scroll(
    image_path: str,
    screen_size: ScreenSize,
    duration: float,
    fps: int,
    focus_point: Tuple[int, int],
    caption: str,
):
    """Create a single Ken Burns style clip."""
    img_clip = ImageClip(image_path).set_duration(duration)
    zoomed = img_clip.fx(
        crop,
        width=screen_size[0],
        height=screen_size[1],
        x_center=focus_point[0],
        y_center=focus_point[1],
    )
    return overlay_caption(zoomed, caption, screen_size)


def make_filmstrip(input_folder: str) -> str:
    """Build final video from assets in *input_folder*."""
    image_files = sorted(
        f
        for f in glob.glob(os.path.join(input_folder, "*"))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not image_files:
        raise FileNotFoundError("No images found in input folder")

    audio_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ]
    if not audio_files:
        raise FileNotFoundError("No audio file found in input folder")
    audio_path = os.path.join(input_folder, audio_files[0])

    beat_times = extract_beats(audio_path)

    clips: List[CompositeVideoClip] = []
    for i, path in enumerate(image_files):
        caption = extract_caption(path)
        img = Image.open(path)
        focus_point = detect_focus_point(img)
        t0 = beat_times[i] if i < len(beat_times) else beat_times[-1]
        t1 = beat_times[i + 1] if i + 1 < len(beat_times) else t0 + 0.6
        duration = t1 - t0
        clip = ken_burns_scroll(
            path, (1080, 1920), duration, 30, focus_point, caption
        )
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_audio(AudioFileClip(audio_path))
    output_path = os.path.join(input_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=30, codec="libx264")
    return output_path
