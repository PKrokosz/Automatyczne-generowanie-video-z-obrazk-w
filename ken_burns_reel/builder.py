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

# Panel camera imports
import numpy as np
from .panels import detect_panels, order_panels_lr_tb
import cv2

def _fit_window_to_box(img_w, img_h, box, target_size):
    """Return (cx, cy, win_w, win_h) framing window to fit a panel."""
    x, y, w, h = box
    tw, th = target_size
    scale_w = w / tw
    scale_h = h / th
    scale = max(scale_w, scale_h)
    win_w = int(tw * scale)
    win_h = int(th * scale)
    cx = x + w // 2
    cy = y + h // 2
    left = max(0, min(cx - win_w // 2, img_w - win_w))
    top = max(0, min(cy - win_h // 2, img_h - win_h))
    return (left + win_w // 2, top + win_h // 2, win_w, win_h)


def _interp(a, b, t):
    return a + (b - a) * t


def make_panels_cam_clip(
    image_path: str,
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 1.0,
    travel: float = 0.6,
):
    """Animate camera between comic panels detected in the image."""
    img = Image.open(image_path)
    W, H = img.size
    boxes = order_panels_lr_tb(detect_panels(img))
    if not boxes:
        return ImageClip(np.array(img)).resize(newsize=target_size).set_duration(3)

    base = ImageClip(np.array(img)).set_duration(1).set_fps(fps)
    segs = []
    for i in range(len(boxes)):
        segs.append(("dwell", i))
        if i < len(boxes) - 1:
            segs.append(("travel", (i, i + 1)))
    total = len(boxes) * dwell + (len(boxes) - 1) * travel

    def make_frame(t):
        acc = 0.0
        for kind, payload in segs:
            dur = dwell if kind == "dwell" else travel
            if t <= acc + dur + 1e-6:
                if kind == "dwell":
                    idx = payload
                    cx, cy, ww, wh = _fit_window_to_box(W, H, boxes[idx], target_size)
                    s = 1.02 - 0.02 * (t - acc) / max(1e-6, dur)
                    ww2, wh2 = int(ww / s), int(wh / s)
                    left = int(cx - ww2 // 2)
                    top = int(cy - wh2 // 2)
                    frame = base.get_frame(0)
                    crop = frame[top : top + wh2, left : left + ww2]
                    return cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)[
                        :, :, ::-1
                    ]
                else:
                    i0, i1 = payload
                    c0x, c0y, w0, h0 = _fit_window_to_box(
                        W, H, boxes[i0], target_size
                    )
                    c1x, c1y, w1, h1 = _fit_window_to_box(
                        W, H, boxes[i1], target_size
                    )
                    tau = (t - acc) / max(1e-6, dur)
                    cx = int(_interp(c0x, c1x, tau))
                    cy = int(_interp(c0y, c1y, tau))
                    ww = int(_interp(w0, w1, tau))
                    wh = int(_interp(h0, h1, tau))
                    frame = base.get_frame(0)
                    left = int(cx - ww // 2)
                    top = int(cy - wh // 2)
                    left = max(0, min(left, W - ww))
                    top = max(0, min(top, H - wh))
                    crop = frame[top : top + wh, left : left + ww]
                    return cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)[
                        :, :, ::-1
                    ]
            acc += dur
        return base.get_frame(0)

    from moviepy.video.VideoClip import VideoClip

    anim = VideoClip(make_frame=make_frame, duration=total).set_fps(fps)
    return anim


def make_panels_cam_sequence(
    image_paths: List[str],
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 1.0,
    travel: float = 0.6,
    xfade: float = 0.4,
):
    """
    Buduje jeden film, sklejając panel-camera clippy dla wszystkich stron.
    """
    if not image_paths:
        raise ValueError("make_panels_cam_sequence: empty image_paths")

    clips = [
        make_panels_cam_clip(p, target_size=target_size, fps=fps, dwell=dwell, travel=travel)
        for p in image_paths
    ]
    # Crossfade między stronami
    final = concatenate_videoclips(clips, method="compose", padding=-xfade)
    return final


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
