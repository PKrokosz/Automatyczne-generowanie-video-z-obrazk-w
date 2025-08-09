"""Core building logic for the Ken Burns reel."""
from __future__ import annotations

import glob
import math
import os
from typing import List, Tuple

try:
    from moviepy.editor import (
        AudioFileClip,
        ImageClip,
        CompositeVideoClip,
        VideoClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy import (
        AudioFileClip,
        ImageClip,
        CompositeVideoClip,
        VideoClip,
        concatenate_audioclips,
        concatenate_videoclips,
    )
from moviepy.audio.AudioClip import AudioClip
try:
    from moviepy.audio.fx.audio_fadein import audio_fadein
    from moviepy.audio.fx.audio_fadeout import audio_fadeout
except ImportError:  # moviepy >=2.0
    from moviepy.audio.fx import AudioFadeIn as audio_fadein, AudioFadeOut as audio_fadeout
try:
    from moviepy.video.fx.all import crop
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy.video.fx import Crop as crop
from PIL import Image

from .audio import extract_beats
from .focus import detect_focus_point
from .ocr import extract_caption, text_boxes_stats, page_ocr_data
from .captions import overlay_caption
from .config import IMAGE_EXTS, AUDIO_EXTS

# Panel camera imports
import numpy as np
from .panels import detect_panels, order_panels_lr_tb
import cv2


def _fit_audio_clip(path: str, duration: float, mode: str) -> AudioFileClip:
    """Return an audio clip resized to *duration* using *mode* strategy."""
    audio = AudioFileClip(path)

    def _make_silence(seconds: float) -> AudioClip:
        # Uwaga: t może być skalarem (float) albo wektorem (ndarray).
        def _silence_make_frame(t):
            if np.isscalar(t):
                # zwróć (nchannels,) dla skalarnego czasu
                return np.zeros((audio.nchannels,), dtype=np.float32)
            # zwróć (len(t), nchannels) dla wektora czasu
            return np.zeros((len(t), audio.nchannels), dtype=np.float32)

        return AudioClip(_silence_make_frame, duration=seconds, fps=audio.fps)

    if mode == "trim":
        # Jeśli audio krótsze niż wideo → dopełnij ciszą, żeby fadein/fadeout
        # nie czytały poza EOF.
        if audio.duration >= duration:
            audio = audio.subclip(0, duration)
        else:
            deficit = duration - audio.duration
            audio = concatenate_audioclips([audio, _make_silence(deficit)])

    elif mode == "silence":
        if audio.duration < duration:
            deficit = duration - audio.duration
            audio = concatenate_audioclips([audio, _make_silence(deficit)])
        else:
            audio = audio.subclip(0, duration)

    elif mode == "loop":
        if audio.duration >= duration:
            audio = audio.subclip(0, duration)
        else:
            reps = int(duration // audio.duration)
            rem = duration - reps * audio.duration
            clips = [audio] * reps
            if rem > 0:
                clips.append(audio.subclip(0, rem))
            audio = concatenate_audioclips(clips)

    else:
        raise ValueError(f"unknown audio-fit mode: {mode}")

    audio = audio.set_duration(duration)
    audio = audio_fadein(audio, 0.15)
    audio = audio_fadeout(audio, 0.15)
    return audio

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
        gamma = max(0.7, min(1.4, gamma))
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
    dwell_mode: str = "first",
):
    """Animate camera between comic panels detected in the image."""
    with Image.open(image_path) as im:
        W, H = im.size
        arr = apply_clahe_rgb(np.array(im))
        boxes = order_panels_lr_tb(detect_panels(Image.fromarray(arr)))
        page_ocr = page_ocr_data(Image.fromarray(arr))
    if not boxes:
        return ImageClip(arr).resize(newsize=target_size).set_duration(3)

    # panel weights for dwell time
    weights = []
    stats_cache: List[dict] = []
    for x, y, w, h in boxes:
        area = w * h
        crop_arr = arr[y : y + h, x : x + w]
        gray = cv2.cvtColor(crop_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        # gather stats from cached OCR data
        heights = []
        words = 0
        for bx, by, bw, bh, txt in zip(
            page_ocr.get("left", []),
            page_ocr.get("top", []),
            page_ocr.get("width", []),
            page_ocr.get("height", []),
            page_ocr.get("text", []),
        ):
            if not txt.strip():
                continue
            if bx < x or by < y or bx + bw > x + w or by + bh > y + h:
                continue
            if bh < 4 or bh > 0.5 * h:
                continue
            heights.append(bh)
            words += 1
        med = float(np.median(heights)) * 0.7 if heights else 0.0
        stats_cache.append({"median_word_height": med, "word_count": words})
        has_text = words > 0
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
    segs: List[Tuple[str, int | Tuple[int, int], float]] = []
    if dwell_mode == "each":
        for i in range(len(boxes)):
            segs.append(("dwell", i, dwell_times[i]))
            if i < len(boxes) - 1:
                segs.append(("travel", (i, i + 1), travel))
    else:
        segs.append(("dwell", 0, dwell_times[0]))
        for i in range(len(boxes) - 1):
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
                    stats = stats_cache[idx]
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
                    # Monotoniczny ease-in (rosnąca prędkość po czasie) dla travel:
                    tau_e = (tau * tau) if easing == "ease" else tau
                    # Stały rozmiar kadru w travel – tylko translacja:
                    ww = int((w0 + w1) * 0.5)
                    wh = int((h0 + h1) * 0.5)
                    cx = int(_interp(c0x, c1x, tau_e))
                    cy = int(_interp(c0y, c1y, tau_e))
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
    audio_path: str | None = None,
    audio_fit: str = "trim",
    dwell_mode: str = "first",
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
            dwell_mode=dwell_mode,
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
            if abs(delta) <= 0.08 and clips[i].duration - delta >= 0.2:
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
    if audio_path:
        audio = _fit_audio_clip(audio_path, final_duration, audio_fit)
        final = final.set_audio(audio)
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


def make_filmstrip(input_folder: str, audio_fit: str = "trim") -> str:
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
    video_duration = sum(c.duration for c in clips)
    audio = _fit_audio_clip(audio_path, video_duration, audio_fit)
    final_clip = final_clip.set_audio(audio)
    output_path = os.path.join(input_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=30, codec="libx264")
    return output_path
