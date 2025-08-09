"""Core building logic for the Ken Burns reel."""
from __future__ import annotations

import glob
import math
import os
from typing import List, Tuple

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    CompositeVideoClip,
    concatenate_videoclips,
    VideoClip,
)
from moviepy.audio.fx import audio_fadein, audio_fadeout
from moviepy.video.fx.all import crop
from PIL import Image

from .audio import extract_beats
from .focus import detect_focus_point
from .ocr import extract_caption, text_boxes_stats
from .captions import overlay_caption
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


def ease_in_out(t: float) -> float:
    """Cosine ease-in-out for t in [0,1]."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def apply_clahe_rgb(arr):
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, a, b])
    arr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    # gamma match
    Lf = L2.astype(np.float32) / 255.0
    mean = float(Lf.mean())
    target = 0.55
    if mean > 0:
        gamma = math.log(target) / math.log(mean)
        Lg = np.clip((Lf ** gamma) * 255.0, 0, 255).astype(np.uint8)
        lab2 = cv2.merge([Lg, a, b])
        arr2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return arr2


def make_panels_cam_clip(
    image_path: str,
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 1.0,
    travel: float = 0.6,
    settle: float = 0.1,
    easing: str = "ease",
    dwell_scale: float = 1.0,
):
    """Animate camera between comic panels detected in the image."""
    with Image.open(image_path) as im:
        W, H = im.size
        arr = apply_clahe_rgb(np.array(im))
        boxes = order_panels_lr_tb(detect_panels(Image.fromarray(arr)))
    if not boxes:
        return ImageClip(arr).resize(newsize=target_size).set_duration(3)

    # panel weights for dwell time
    weights = []
    for x, y, w, h in boxes:
        area = w * h
        crop_arr = arr[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        stats = text_boxes_stats(Image.fromarray(crop_arr))
        has_text = stats.get("word_count", 0) > 0
        weight = area * (1 + edge_density) * (1.2 if has_text else 1.0)
        weights.append(weight)

    def normalize_weights(ws, lo=0.7, hi=1.3):
        wmin, wmax = min(ws), max(ws)
        if wmax <= wmin:
            return [1.0] * len(ws)
        return [lo + (w - wmin) * (hi - lo) / (wmax - wmin) for w in ws]

    norm_w = normalize_weights(weights)
    dwell_times = [dwell * w * dwell_scale for w in norm_w]

    base = ImageClip(arr).set_duration(1).set_fps(fps)
    segs = []
    for i in range(len(boxes)):
        segs.append(("dwell", i, dwell_times[i]))
        if i < len(boxes) - 1:
            segs.append(("travel", (i, i + 1), travel))
    total = sum(d for _, _, d in segs)

    def make_frame(t):
        acc = 0.0
        for kind, payload, dur in segs:
            if t <= acc + dur + 1e-6:
                if kind == "dwell":
                    idx = payload
                    cx, cy, ww, wh = _fit_window_to_box(W, H, boxes[idx], target_size)
                    # additional zoom if text is tiny
                    x, y, bw, bh = boxes[idx]
                    stats = text_boxes_stats(Image.fromarray(arr[y : y + bh, x : x + bw]))
                    med = stats.get("median_word_height", 0.0)
                    if med and med / max(1, bh) < 0.035:
                        zoom = 1 / 1.06
                        pad = 24
                        min_w = bw + pad * 2
                        min_h = bh + pad * 2
                        if ww > min_w:
                            ww = max(int(ww * zoom), min_w)
                        if wh > min_h:
                            wh = max(int(wh * zoom), min_h)
                    active = max(0.0, dur - settle)
                    elapsed = t - acc
                    if elapsed <= active:
                        s = 1.02 - 0.02 * elapsed / max(1e-6, active)
                    else:
                        s = 1.0
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
                    tau_e = ease_in_out(tau) if easing == "ease" else tau
                    cx = int(_interp(c0x, c1x, tau_e))
                    cy = int(_interp(c0y, c1y, tau_e))
                    ww = int(_interp(w0, w1, tau_e))
                    wh = int(_interp(h0, h1, tau_e))
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
    settle: float = 0.1,
    easing: str = "ease",
    dwell_scale: float = 1.0,
    align_beat: bool = False,
    beat_times: List[float] | None = None,
):
    """
    Buduje jeden film, sklejając panel-camera clippy dla wszystkich stron.
    """
    if not image_paths:
        raise ValueError("make_panels_cam_sequence: empty image_paths")

    clips = [
        make_panels_cam_clip(
            p,
            target_size=target_size,
            fps=fps,
            dwell=dwell,
            travel=travel,
            settle=settle,
            easing=easing,
            dwell_scale=dwell_scale,
        )
        for p in image_paths
    ]

    # prepare start times
    starts = [0.0]
    for i in range(1, len(clips)):
        start = starts[i - 1] + clips[i - 1].duration - xfade
        if align_beat and beat_times:
            nearest = min(beat_times, key=lambda b: abs(b - start))
            delta = nearest - start
            if abs(delta) <= 0.08:
                start = nearest
                clips[i] = clips[i].set_duration(clips[i].duration - delta)
        starts.append(start)

    # apply start times and crossfades
    for i, clip in enumerate(clips):
        clip = clip.set_start(starts[i])
        if i > 0:
            clip = clip.crossfadein(xfade)
        clips[i] = clip

    # audio fades for crossfade
    for i in range(len(clips) - 1):
        clips[i] = clips[i].audio_fadeout(0.15)
        clips[i + 1] = clips[i + 1].audio_fadein(0.15)

    final_duration = starts[-1] + clips[-1].duration
    final = CompositeVideoClip(clips, size=target_size).set_duration(final_duration)
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
        with Image.open(path) as img:
            focus_point = detect_focus_point(img)
        t0 = beat_times[i] if i < len(beat_times) else beat_times[-1]
        t1 = beat_times[i + 1] if i + 1 < len(beat_times) else t0 + 0.6
        duration = t1 - t0
        clip = ken_burns_scroll(
            path, (1080, 1920), duration, 30, focus_point, caption
        )
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    audioclip = AudioFileClip(audio_path)
    audioclip = audio_fadein.audio_fadein(audioclip, 0.15)
    audioclip = audio_fadeout.audio_fadeout(audioclip, 0.15)
    final_clip = final_clip.set_audio(audioclip)
    output_path = os.path.join(input_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=30, codec="libx264")
    return output_path
