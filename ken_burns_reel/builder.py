"""Core building logic for the Ken Burns reel."""
from __future__ import annotations

import glob
import math
import os
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
from .transitions import (
    slide_transition,
    smear_transition,
    whip_pan_transition,
    smear_bg_crossfade_fg,
)

# Panel camera imports
import numpy as np
from .panels import detect_panels, order_panels_lr_tb, alpha_bbox
import cv2
from .utils import gaussian_blur, _set_fps


def _fit_audio_clip(path: str, duration: float, mode: str, gain_db: float = 0.0) -> AudioFileClip:
    """Return an audio clip resized to *duration* using *mode* strategy.

    Parameters
    ----------
    path: str
        Audio file path.
    duration: float
        Desired duration in seconds.
    mode: str
        "trim", "silence", or "loop".
    gain_db: float
        Optional gain in decibels applied to the resulting clip.
    """
    duration = max(0.0, duration)
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
    if gain_db:
        audio = audio.volumex(10 ** (gain_db / 20.0))
    audio = audio_fadein(audio, 0.12)
    audio = audio_fadeout(audio, 0.12)
    return audio

def _fit_window_to_box(img_w, img_h, box, target_size, bleed: int = 0):
    """Return (cx, cy, win_w, win_h) framing window to fit a panel."""
    x, y, w, h = box
    if bleed:
        x = max(0, x - bleed)
        y = max(0, y - bleed)
        w = min(img_w - x, w + 2 * bleed)
        h = min(img_h - y, h + 2 * bleed)
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


def ease_in(t: float) -> float:
    """Cosine ease-in for t in [0,1]."""
    return 1 - math.cos(0.5 * math.pi * t)


def ease_out(t: float) -> float:
    """Cosine ease-out for t in [0,1]."""
    return math.sin(0.5 * math.pi * t)


def _apply_witcher_look(frame: np.ndarray, vignette_strength: float) -> np.ndarray:
    rng = np.random.default_rng()
    f = frame.astype(np.float32)
    M = np.array(
        [
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131],
        ],
        dtype=np.float32,
    )
    f = np.clip(f @ M.T, 0, 255)
    f += rng.normal(0.0, 8.0, f.shape[:2])[..., None].astype(np.float32)
    f = np.clip(f, 0, 255)
    if vignette_strength > 0:
        H, W = f.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
        d /= np.hypot(W / 2, H / 2) + 1e-6
        f *= (1 - 1.5 * vignette_strength * d)[..., None]
    return np.clip(f, 0, 255).astype(np.uint8)


def _get_ease_fn(name: str):
    return {
        "linear": lambda t: t,
        "in": ease_in,
        "out": ease_out,
        "inout": ease_in_out,
    }.get(name, ease_in_out)


def _with_duration(clip: VideoClip, duration: float) -> VideoClip:
    """Compat helper for moviepy 1.x/2.x set_duration API."""
    if hasattr(clip, "set_duration"):
        return clip.set_duration(duration)
    return clip.with_duration(duration)


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


def enhance_panel(arr: np.ndarray) -> np.ndarray:
    """Enhance RGBA panel; process RGB and keep alpha."""
    if arr.shape[-1] < 4:
        rgb = apply_clahe_rgb(arr)
        return rgb
    rgb = apply_clahe_rgb(arr[:, :, :3])
    alpha = arr[:, :, 3:4]
    return np.concatenate([rgb, alpha], axis=2)


def _paste_rgba_clipped(canvas: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
    """Safely paste *overlay* onto *canvas* with clipping.

    Both arrays are expected in RGBA format. The overlay may extend beyond the
    canvas bounds; only the intersecting region will be copied.
    """
    H, W = canvas.shape[:2]
    h, w = overlay.shape[:2]

    sx, sy = x, y
    ex, ey = x + w, y + h

    dst_x0 = max(0, sx)
    dst_y0 = max(0, sy)
    dst_x1 = min(W, ex)
    dst_y1 = min(H, ey)

    src_x0 = max(0, -sx)
    src_y0 = max(0, -sy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if (dst_x1 <= dst_x0) or (dst_y1 <= dst_y0):
        return
    ov = overlay[src_y0:src_y1, src_x0:src_x1]
    rgb = ov[:, :, :3].astype(np.uint16)
    alpha = ov[:, :, 3:4].astype(np.uint16)
    rgb = (rgb * alpha // 255).astype(np.uint8)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1, :3] = rgb
    canvas[dst_y0:dst_y1, dst_x0:dst_x1, 3] = ov[:, :, 3]


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    """Convert ``"#rrggbb"`` hex color to an RGB tuple."""
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"invalid hex color: {value}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return r, g, b


def _zoom_image_center(img: np.ndarray, scale: float) -> np.ndarray:
    """Zoom ``img`` around its centre by ``scale`` keeping original size."""
    if abs(scale - 1.0) < 1e-6:
        return img
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    if scale >= 1.0:
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        return resized[y0 : y0 + h, x0 : x0 + w]
    canvas = np.zeros_like(img)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _darken_region_with_alpha_clipped(
    canvas_rgb: np.ndarray, alpha_map: np.ndarray, x: int, y: int, strength: float
) -> None:
    """Darken *canvas_rgb* using *alpha_map* placed at (x, y) with clipping."""
    H, W = canvas_rgb.shape[:2]
    h, w = alpha_map.shape[:2]

    sx, sy = x, y
    ex, ey = x + w, y + h

    dst_x0 = max(0, sx)
    dst_y0 = max(0, sy)
    dst_x1 = min(W, ex)
    dst_y1 = min(H, ey)

    src_x0 = max(0, -sx)
    src_y0 = max(0, -sy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    if (dst_x1 <= dst_x0) or (dst_y1 <= dst_y0):
        return

    region = canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :].astype(np.float32)
    a = alpha_map[src_y0:src_y1, src_x0:src_x1].astype(np.float32) / 255.0
    mult = (1.0 - strength * a)[..., None]
    canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :] = np.clip(region * mult, 0, 255).astype(
        np.uint8
    )


def _add_rgb_clipped(canvas_rgb: np.ndarray, rgb: np.ndarray, x: int, y: int) -> None:
    """Add *rgb* onto *canvas_rgb* with clipping."""
    H, W = canvas_rgb.shape[:2]
    h, w = rgb.shape[:2]
    sx, sy = x, y
    ex, ey = x + w, y + h
    dst_x0 = max(0, sx)
    dst_y0 = max(0, sy)
    dst_x1 = min(W, ex)
    dst_y1 = min(H, ey)
    src_x0 = max(0, -sx)
    src_y0 = max(0, -sy)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return
    dst = canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :].astype(np.float32)
    src = rgb[src_y0:src_y1, src_x0:src_x1]
    canvas_rgb[dst_y0:dst_y1, dst_x0:dst_x1, :] = np.clip(dst + src, 0, 255).astype(
        np.uint8
    )


def _attach_mask(clip: VideoClip, mask: VideoClip) -> VideoClip:
    """Attach *mask* to *clip* using API compatible with moviepy v1/v2."""
    if hasattr(clip, "with_mask"):
        return clip.with_mask(mask)
    return clip.set_mask(mask)


def _make_underlay(arr: np.ndarray, target_size: Tuple[int, int], mode: str) -> np.ndarray:
    """Generate a background underlay for the page based on *mode*.

    Parameters
    ----------
    arr:
        Source image array.
    target_size:
        Desired output size ``(width, height)``.
    mode:
        One of ``"none"``, ``"stretch"``, ``"gradient"`` or ``"blur"``.
    """
    W, H = target_size
    if mode == "none":
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:] = (8, 10, 14)
    elif mode == "stretch":
        canvas = cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC)
        canvas = np.clip(canvas.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    elif mode == "gradient":
        y = np.linspace(0, 1, H)[:, None]
        top = np.array([20, 25, 30], dtype=np.float32)
        bottom = np.array([60, 60, 60], dtype=np.float32)
        canvas = (top + (bottom - top) * y).astype(np.uint8)
        canvas = np.repeat(canvas, W, axis=1)
    else:  # blur
        canvas = cv2.resize(arr, (W, H), interpolation=cv2.INTER_CUBIC)
        limit = min(W, H)
        k = 51 if limit >= 51 else (limit // 2 * 2 + 1)
        canvas = gaussian_blur(canvas, (k, k))
        hsv = cv2.cvtColor(canvas, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.6
        hsv[:, :, 2] *= 0.7
        canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    dist /= dist.max() + 1e-6
    vign = 1 - 0.15 * dist
    canvas = (canvas.astype(np.float32) * vign[..., None]).astype(np.uint8)
    return canvas


def make_panels_cam_clip(
    image_path: str,
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 1.0,
    travel: float = 0.6,
    settle: float = 0.14,
    travel_ease: str = "inout",
    dwell_scale: float = 1.0,
    dwell_mode: str = "first",
    bg_mode: str = "blur",
    page_scale: float = 0.92,
    bg_parallax: float = 0.85,
    panel_bleed: int = 24,
    zoom_max: float = 1.06,
    easing: str | None = None,
):
    """Animate camera between comic panels detected in the image.

    Parameters
    ----------
    easing:
        Backwards compatible alias for ``travel_ease``. Passing ``"ease"`` maps
        to ``"inout"``.
    """
    if easing is not None:
        travel_ease = "inout" if easing == "ease" else easing
    with Image.open(image_path) as im:
        W, H = im.size
        arr = apply_clahe_rgb(np.array(im))
        boxes = order_panels_lr_tb(detect_panels(Image.fromarray(arr)))
        page_ocr = page_ocr_data(Image.fromarray(arr))
    if not boxes:
        return _set_fps(ImageClip(arr).resize(newsize=target_size).set_duration(3), fps)

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

    base = _set_fps(ImageClip(arr).set_duration(1), fps)
    underlay0 = _make_underlay(arr, target_size, bg_mode)
    Wout, Hout = target_size
    fw = int(Wout * page_scale)
    fh = int(Hout * page_scale)
    x0 = (Wout - fw) // 2
    y0 = (Hout - fh) // 2
    page_layer0 = None
    if page_scale < 1.0:
        page_layer0 = cv2.resize(arr, (fw, fh), interpolation=cv2.INTER_CUBIC)
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
                    cx, cy, ww, wh = _fit_window_to_box(
                        W, H, boxes[idx], target_size, panel_bleed
                    )
                    # additional zoom if text is tiny
                    x, y, bw, bh = boxes[idx]
                    stats = stats_cache[idx]
                    med = stats.get("median_word_height", 0.0)
                    if med and med / max(1, bh) < 0.035:
                        zoom = 1 / zoom_max
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
                    fg = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_CUBIC)
                    bg = underlay0
                    canvas = bg.copy()
                    if page_layer0 is not None:
                        pg_canvas = np.zeros_like(bg)
                        pg_canvas[y0 : y0 + fh, x0 : x0 + fw] = page_layer0
                        mask = pg_canvas.sum(axis=2) > 0
                        canvas[mask] = pg_canvas[mask]
                    canvas[y0 : y0 + fh, x0 : x0 + fw] = fg
                    return canvas[:, :, ::-1]
                else:
                    i0, i1 = payload
                    c0x, c0y, w0, h0 = _fit_window_to_box(
                        W, H, boxes[i0], target_size, panel_bleed
                    )
                    c1x, c1y, w1, h1 = _fit_window_to_box(
                        W, H, boxes[i1], target_size, panel_bleed
                    )
                    tau = (t - acc) / max(1e-6, dur)
                    if travel_ease == "in":
                        tau_e = tau * tau
                    elif travel_ease == "out":
                        tau_e = 1 - (1 - tau) * (1 - tau)
                    elif travel_ease == "inout":
                        tau_e = ease_in_out(tau)
                    else:
                        tau_e = tau
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
                    fg = cv2.resize(crop, (fw, fh), interpolation=cv2.INTER_CUBIC)
                    dx = int((c1x - c0x) * tau_e * (1 - bg_parallax))
                    dy = int((c1y - c0y) * tau_e * (1 - bg_parallax))
                    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
                    bg = cv2.warpAffine(underlay0, M, (Wout, Hout), borderMode=cv2.BORDER_REFLECT)
                    canvas = bg.copy()
                    if page_layer0 is not None:
                        pg_canvas = np.zeros_like(bg)
                        pg_canvas[y0 : y0 + fh, x0 : x0 + fw] = page_layer0
                        pg_parallax = (bg_parallax + 1.0) / 2.0
                        dxp = int((c1x - c0x) * tau_e * (1 - pg_parallax))
                        dyp = int((c1y - c0y) * tau_e * (1 - pg_parallax))
                        Mp = np.float32([[1, 0, -dxp], [0, 1, -dyp]])
                        pg_canvas = cv2.warpAffine(
                            pg_canvas, Mp, (Wout, Hout), borderMode=cv2.BORDER_REFLECT
                        )
                        mask = pg_canvas.sum(axis=2) > 0
                        canvas[mask] = pg_canvas[mask]
                    canvas[y0 : y0 + fh, x0 : x0 + fw] = fg
                    return canvas[:, :, ::-1]
            acc += dur
        return base.get_frame(0)

    from moviepy.video.VideoClip import VideoClip

    anim = _set_fps(VideoClip(make_frame, duration=total), fps)
    return anim


def make_panels_cam_sequence(
    image_paths: List[str],
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 1.0,
    travel: float = 0.6,
    xfade: float = 0.4,
    settle: float = 0.14,
    travel_ease: str = "inout",
    dwell_scale: float = 1.0,
    align_beat: bool = False,
    beat_times: List[float] | None = None,
    audio_path: str | None = None,
    audio_fit: str = "trim",
    dwell_mode: str = "first",
    bg_mode: str = "blur",
    page_scale: float = 0.92,
    bg_parallax: float = 0.85,
    panel_bleed: int = 24,
    zoom_max: float = 1.06,
    easing: str | None = None,
):
    """
    Buduje jeden film, sklejając panel-camera clippy dla wszystkich stron.
    """
    if not image_paths:
        raise ValueError("make_panels_cam_sequence: empty image_paths")

    if easing is not None:
        travel_ease = "inout" if easing == "ease" else easing

    clips = [
        make_panels_cam_clip(
            p,
            target_size=target_size,
            fps=fps,
            dwell=dwell,
            travel=travel,
            settle=settle,
            travel_ease=travel_ease,
            dwell_scale=dwell_scale,
            dwell_mode=dwell_mode,
            bg_mode=bg_mode,
            page_scale=page_scale,
            bg_parallax=bg_parallax,
            panel_bleed=panel_bleed,
            zoom_max=zoom_max,
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
    return _set_fps(final, fps)


def make_panels_items_sequence(
    panel_paths: List[str],
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 0.7,
    trans: str = "smear",
    trans_dur: float = 0.3,
    smear_strength: float = 1.0,
    zoom_max: float = 1.06,
    page_scale: float = 0.92,
    bg_mode: str = "blur",
    bg_parallax: float = 0.85,
) -> CompositeVideoClip:
    """Build a sequence from pre-cropped panel images."""

    if not panel_paths:
        raise ValueError("make_panels_items_sequence: empty panel_paths")

    Wout, Hout = target_size
    fw = int(Wout * page_scale)
    fh = int(Hout * page_scale)
    x0 = (Wout - fw) // 2
    y0 = (Hout - fh) // 2

    def _panel_clip(path: str) -> VideoClip:
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))
        underlay = _make_underlay(arr, target_size, bg_mode)
        ah, aw = arr.shape[:2]
        scale0 = max(fw / aw, fh / ah)
        base = cv2.resize(
            arr, (int(aw * scale0), int(ah * scale0)), interpolation=cv2.INTER_CUBIC
        )

        def make_frame(t: float):
            p = min(1.0, t / max(1e-6, dwell))
            s = 1 + (zoom_max - 1) * p
            w = int(base.shape[1] * s)
            h = int(base.shape[0] * s)
            resized = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
            left = (w - fw) // 2
            top = (h - fh) // 2
            fg = resized[top : top + fh, left : left + fw]
            canvas = underlay.copy()
            canvas[y0 : y0 + fh, x0 : x0 + fw] = fg
            return canvas[:, :, ::-1]

        return _set_fps(VideoClip(make_frame, duration=dwell + trans_dur), fps)

    full_clips = [_panel_clip(p) for p in panel_paths]

    seq: List[VideoClip] = []
    for i, clip in enumerate(full_clips):
        seq.append(clip.subclip(0, dwell))
        if i < len(full_clips) - 1:
            vec = (0.0, 0.0)
            nxt = full_clips[i + 1]
            if trans == "slide":
                tclip = slide_transition(clip, nxt, trans_dur, target_size, fps)
            elif trans == "xfade":
                tail = clip.subclip(dwell - trans_dur, dwell)
                head = nxt.subclip(0, trans_dur)
                tclip = (
                    _set_fps(
                        CompositeVideoClip(
                            [tail.crossfadeout(trans_dur), head.crossfadein(trans_dur)],
                            size=target_size,
                        ).set_duration(trans_dur),
                        fps,
                    )
                )
            elif trans == "whip":
                tclip = whip_pan_transition(clip, nxt, trans_dur, target_size, vec, fps=fps)
            else:  # smear
                tclip = smear_transition(
                    clip,
                    nxt,
                    trans_dur,
                    target_size,
                    vec,
                    strength=smear_strength,
                    fps=fps,
                )
            seq.append(tclip)

    final = concatenate_videoclips(seq, method="compose")
    return _set_fps(final, fps)


def compute_segment_timing(
    bpm: int | None,
    beats_per_panel: float,
    beats_travel: float,
    readability_ms: int,
    min_dwell: float,
    max_dwell: float,
    settle_min: float,
    settle_max: float,
) -> tuple[float, float, float]:
    dwell_base = max(readability_ms / 1000.0, min_dwell)
    if bpm:
        beat = 60.0 / bpm
        dwell = max(dwell_base, beats_per_panel * beat)
        dwell = min(dwell, max_dwell)
        travel = beats_travel * beat
        travel = max(0.32, min(0.55, travel))
    else:
        dwell = dwell_base
        travel = 0.4
    settle = max(settle_min, min(settle_max, (settle_min + settle_max) / 2.0))
    return dwell, travel, settle


def make_panels_overlay_sequence(
    page_paths: List[str],
    panels_dir: str,
    target_size=(1080, 1920),
    fps: int = 30,
    dwell: float = 0.9,
    travel: float = 0.5,
    settle: float = 0.14,
    travel_ease: str = "inout",
    align_beat: bool = False,
    beat_times=None,
    overlay_fit: float = 0.7,
    overlay_margin: int = 12,
    overlay_mode: str = "anchored",
    overlay_scale: float = 1.6,
    bg_source: str = "page",
    bg_blur: float = 8.0,
    bg_tex: str = "vignette",
    bg_tone_strength: float = 0.7,
    parallax_bg: float = 0.05,
    parallax_fg: float = 0.0,
    fg_shadow: float = 0.25,
    fg_shadow_blur: int = 18,
    fg_shadow_offset: int = 4,
    fg_shadow_mode: str = "soft",
    deep_bg_mode: str = "gradient",
    deep_bg_parallax: float = 0.02,
    page_desaturate: float = 0.15,
    page_dim: float = 0.15,
    fg_glow: float = 0.10,
    fg_glow_blur: int = 24,
    overlay_edge: str = "feather",
    overlay_edge_strength: float = 0.6,
    min_panel_area_ratio: float = 0.03,
    gutter_thicken: int = 0,
    debug_overlay: bool = False,
    page_scale_overlay: float = 1.0,
    bg_vignette: float = 0.15,
    overlay_pop: float = 1.0,
    overlay_jitter: float = 0.0,
    overlay_frame_px: int = 0,
    overlay_frame_color: str = "#000000",
    bg_offset: float = 0.0,
    fg_offset: float = 0.0,
    bg_drift_zoom: float = 0.0,
    bg_drift_speed: float = 0.0,
    fg_drift_zoom: float = 0.0,
    fg_drift_speed: float = 0.0,
    travel_path: str = "linear",
    deep_bottom_glow: float = 0.0,
    look: str = "none",
    timing_profile: str = "free",
    bpm: int | None = None,
    beats_per_panel: float = 2.0,
    beats_travel: float = 0.5,
    readability_ms: int = 900,
    min_dwell: float = 1.0,
    max_dwell: float = 1.8,
    settle_min: float = 0.12,
    settle_max: float = 0.22,
    quantize: str = "off",
    limit_items: int = 999,
    trans: str = "smear",
    trans_dur: float = 0.3,
    smear_strength: float = 1.0,
) -> CompositeVideoClip:
    """Render overlay sequence with static foreground panels."""

    if not page_paths:
        raise ValueError("make_panels_overlay_sequence: empty page_paths")

    items = []
    for idx, p in enumerate(page_paths, 1):
        with Image.open(p) as im:
            page_arr = np.array(im.convert("RGB"))
            boxes = order_panels_lr_tb(
                detect_panels(im, min_panel_area_ratio, gutter_thicken)
            )
        Hpage, Wpage = page_arr.shape[:2]
        cx_pg, cy_pg, win_w_pg, win_h_pg = _fit_window_to_box(
            Wpage, Hpage, (0, 0, Wpage, Hpage), target_size
        )
        page_frame = (cx_pg, cy_pg, win_w_pg, win_h_pg)
        panel_folder = os.path.join(panels_dir, f"page_{idx:04d}")
        panel_files = sorted(glob.glob(os.path.join(panel_folder, "panel_*.png")))
        for box, panel_file in zip(boxes, panel_files):
            items.append({
                "page": page_arr,
                "panel": panel_file,
                "box": box,
                "frame": page_frame,
            })
            if len(items) >= limit_items:
                break
        if len(items) >= limit_items:
            break

    if not items:
        raise ValueError("make_panels_overlay_sequence: no panels found")

    for it in items:
        with Image.open(it["panel"]) as im:
            it["panel_arr"] = np.array(im.convert("RGBA"))
        x, y, w, h = it["box"]
        it["center"] = (x + w / 2.0, y + h / 2.0)

    Wout, Hout = target_size
    frame_rgb = _hex_to_rgb(overlay_frame_color)

    bg_clips: List[VideoClip] = []
    fg_clips: List[VideoClip] = []

    ease_fn = _get_ease_fn(travel_ease)
    parallax_fg = max(0.0, min(0.5, parallax_fg))
    bg_tone_strength = max(0.0, min(1.0, bg_tone_strength))

    if fg_shadow_mode == "hard":
        fg_shadow_blur = max(1, fg_shadow_blur // 2)
        fg_shadow += 0.05
    fg_shadow = max(0.0, min(0.5, fg_shadow))

    if timing_profile != "free":
        dwell, travel, settle = compute_segment_timing(
            bpm,
            beats_per_panel,
            beats_travel,
            readability_ms,
            min_dwell,
            max_dwell,
            settle_min,
            settle_max,
        )

    # prepare per-segment timing and optional quantization
    @dataclass
    class SegmentTiming:
        """Per-segment timing values and snap offset."""

        start: float
        dwell: float
        travel: float
        settle: float
        snap_delta: float = 0.0

    seg_timings: List[SegmentTiming] = []
    start_t = 0.0
    for _ in items:
        seg_timings.append(SegmentTiming(start_t, dwell, travel, settle))
        start_t += dwell + travel + settle

    # optional snapping to detected beats before grid quantization
    if align_beat and beat_times:
        snapped: List[SegmentTiming] = []
        for s in seg_timings:
            start, d, tr, st = s.start, s.dwell, s.travel, s.settle
            if start == 0.0:
                snapped.append(SegmentTiming(start, d, tr, st, 0.0))
                continue
            nearest = min(beat_times, key=lambda b: abs(b - start))
            delta = nearest - start
            if abs(delta) <= 0.08 and d - delta >= max(0.2, readability_ms / 1000.0):
                start = nearest
                if delta > 0:
                    d = d - delta
                else:
                    st = min(settle_max, st - delta)
            snapped.append(SegmentTiming(start, d, tr, st, delta))
        seg_timings = snapped

    # then optional BPM grid quantization
    if bpm and quantize != "off":
        beat = 60.0 / bpm
        step = beat if quantize == "1/4" else beat / 2.0
        new_timings: List[SegmentTiming] = []
        for s in seg_timings:
            start, d, tr, st = s.start, s.dwell, s.travel, s.settle
            delta = 0.0
            if start != 0:
                q = round(start / step) * step
                delta = q - start
                if abs(delta) <= 0.1:
                    st += delta
                    st = max(0.0, st)
                    st = max(settle_min, min(settle_max, st))
                    start = q
            new_timings.append(SegmentTiming(start, d, tr, st, delta))
        seg_timings = new_timings
        overlay_jitter = 0.0

    timing_log: List[Dict[str, float]] = []
    meta = {"timing_profile": timing_profile, "bpm": bpm}
    timing_log.append(meta)
    for idx, s in enumerate(seg_timings):
        row = {
            "index": idx,
            "start": round(s.start, 3),
            "dwell": round(s.dwell, 3),
            "travel": round(s.travel, 3),
            "settle": round(s.settle, 3),
            "snap_delta": round(s.snap_delta, 3),
        }
        if bpm:
            beat_len = 60.0 / bpm
            row["beat_index"] = round(s.start / beat_len, 3)
        timing_log.append(row)
    try:
        with open("timing_log.jsonl", "w", encoding="utf8") as f:
            for row in timing_log:
                f.write(json.dumps(row) + "\n")
    except OSError:
        pass

    for i, it in enumerate(items):
        s = seg_timings[i]
        start, dwell, travel, settle = s.start, s.dwell, s.travel, s.settle
        page_arr = it["page"]
        panel_arr = it["panel_arr"]
        box = it["box"]
        Hpage, Wpage = page_arr.shape[:2]
        cx0, cy0, win_w, win_h = it["frame"]
        # zapewnij cx1, cy1 także dla center-mode
        if i + 1 < len(items) and items[i + 1]["page"] is page_arr:
            cx1, cy1, _, _ = items[i + 1]["frame"]
        else:
            cx1, cy1 = cx0, cy0
        left0 = int(max(0, min(cx0 - win_w // 2, Wpage - win_w)))
        top0 = int(max(0, min(cy0 - win_h // 2, Hpage - win_h)))
        if i + 1 < len(items) and items[i + 1]["page"] is page_arr:
            dx = items[i + 1]["center"][0] - it["center"][0]
            dy = items[i + 1]["center"][1] - it["center"][1]
            left1 = int(max(0, min(left0 + int(dx), Wpage - win_w)))
            top1 = int(max(0, min(top0 + int(dy), Hpage - win_h)))
        else:
            left1, top1 = left0, top0

        def make_bg(t, arr=page_arr, l0=left0, t0=top0, l1=left1, t1=top1, seg_start=start):
            if bg_source == "page":
                tt = t - bg_offset
                if tt <= dwell:
                    l, tp = l0, t0
                else:
                    p = (tt - dwell) / max(1e-6, travel)
                    p = ease_fn(p)
                    l = l0 + parallax_bg * (l1 - l0) * p
                    tp = t0 + parallax_bg * (t1 - t0) * p
                    if travel_path == "arc":
                        dx = l1 - l0
                        dy = t1 - t0
                        dist = math.hypot(dx, dy)
                        if dist > 1e-6:
                            px, py = -dy / dist, dx / dist
                            off = math.sin(math.pi * p) * dist * 0.25 * parallax_bg
                            l += px * off
                            tp += py * off
                l = int(max(0, min(l, Wpage - win_w)))
                tp = int(max(0, min(tp, Hpage - win_h)))
                crop = arr[tp : tp + win_h, l : l + win_w]
                frame_raw = cv2.resize(crop, (Wout, Hout), interpolation=cv2.INTER_CUBIC)
                if bg_blur > 0:
                    frame_raw = cv2.GaussianBlur(frame_raw, (0, 0), bg_blur)
                mid = frame_raw
                if page_desaturate > 0 or page_dim > 0:
                    hsv = cv2.cvtColor(mid, cv2.COLOR_RGB2HSV).astype(np.float32)
                    if page_desaturate > 0:
                        hsv[:, :, 1] *= 1 - page_desaturate
                    if page_dim > 0:
                        hsv[:, :, 2] *= 1 - page_dim
                    mid = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                if bg_tex == "vignette" and bg_vignette > 0 and look != "witcher1":
                    yy, xx = np.mgrid[0:Hout, 0:Wout]
                    dist = np.sqrt((xx - Wout / 2) ** 2 + (yy - Hout / 2) ** 2)
                    dist /= dist.max() + 1e-6
                    vign = 1 - bg_vignette * dist
                    mid = (mid.astype(np.float32) * vign[..., None]).astype(np.uint8)
                elif bg_tex == "gradient":
                    gx = np.linspace(0.9, 1.0, Wout, dtype=np.float32)[None, :].repeat(Hout, 0)
                    mid = np.clip(mid.astype(np.float32) * gx[..., None], 0, 255).astype(np.uint8)
                # bg_tex == "none" -> bez tekstur
                frame = mid
            else:
                frame = _make_underlay(arr, target_size, bg_source)

            if page_scale_overlay < 1.0:
                fw = int(Wout * page_scale_overlay)
                fh = int(Hout * page_scale_overlay)
                scaled = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_CUBIC)
                canvas = np.zeros_like(frame)
                x0 = (Wout - fw) // 2
                y0 = (Hout - fh) // 2
                canvas[y0 : y0 + fh, x0 : x0 + fw] = scaled
                frame = canvas

            if deep_bg_mode != "none" and travel > 0:
                if deep_bg_mode == "gradient":
                    base = np.linspace(0, 255, Wout, dtype=np.uint8)[None, :].repeat(Hout, 0)
                    deep = np.dstack([base, base, base])
                else:
                    rng = np.random.default_rng(0)
                    deep = rng.integers(0, 256, (Hout, Wout, 3), dtype=np.uint8)
                p = min(1.0, t / max(1e-6, dwell + travel))
                dx = int(deep_bg_parallax * parallax_bg * Wout * p)
                dy = int(deep_bg_parallax * parallax_bg * Hout * p)
                deep = np.roll(np.roll(deep, dx, axis=1), dy, axis=0)
                frame = cv2.addWeighted(deep, 0.25, frame, 0.75, 0)
            if look == "witcher1":
                frame = _apply_witcher_look(frame, bg_vignette)
            if bg_drift_zoom > 0 and bg_drift_speed > 0:
                phase = 2 * math.pi * bg_drift_speed * seg_start
                scale = 1.0 + bg_drift_zoom * math.sin(phase + 2 * math.pi * bg_drift_speed * t)
                frame = _zoom_image_center(frame, scale)
            if deep_bottom_glow > 0:
                grad = np.linspace(1.0, 1.0 + deep_bottom_glow, Hout, dtype=np.float32)[:, None]
                frame = np.clip(frame.astype(np.float32) * grad[..., None], 0, 255).astype(np.uint8)
            return frame

        bg = _set_fps(VideoClip(make_bg, duration=dwell + travel + settle), fps)
        bg_clips.append(bg)

        x0, y0, w0, h0 = alpha_bbox(panel_arr)
        if panel_arr.shape[-1] < 4 or w0 <= 0 or h0 <= 0 or np.all(panel_arr[:, :, 3] == 0):
            x, y, w, h = box
            panel_arr = np.dstack(
                [
                    page_arr[y : y + h, x : x + w],
                    np.full((h, w, 1), 255, dtype=np.uint8),
                ]
            )
            x0, y0, w0, h0 = 0, 0, w, h
        panel = panel_arr[y0 : y0 + h0, x0 : x0 + w0]
        panel = enhance_panel(panel)

        if overlay_mode == "anchored":
            x, y, w, h = box
            S = Wout / win_w
            dst_w = int(round(w * S * overlay_scale))
            dst_h = int(round(h * S * overlay_scale))
            resized = cv2.resize(panel, (dst_w, dst_h), interpolation=cv2.INTER_CUBIC)
            alpha_panel = resized[:, :, 3]
            if overlay_edge == "feather":
                sigma = max(1, int(overlay_edge_strength * 10))
                alpha_panel = gaussian_blur(alpha_panel, sigma=sigma)
            elif overlay_edge == "torn":
                noise = np.random.rand(*alpha_panel.shape).astype(np.float32)
                noise = cv2.GaussianBlur(noise, (0, 0), 5)
                thr = 0.5 + overlay_edge_strength * 0.25
                alpha_panel = (alpha_panel.astype(np.float32) * (noise > thr)).astype(np.uint8)
            if overlay_frame_px > 0:
                k = np.ones((overlay_frame_px * 2 + 1, overlay_frame_px * 2 + 1), np.uint8)
                dil = cv2.dilate(alpha_panel, k, iterations=1)
                ring = cv2.subtract(dil, alpha_panel)
                if np.any(ring):
                    ring_rgba = np.zeros_like(resized)
                    ring_rgba[:, :, :3] = frame_rgb
                    ring_rgba[:, :, 3] = ring
                    canvas = np.zeros_like(resized)
                    _paste_rgba_clipped(canvas, ring_rgba, 0, 0)
                    _paste_rgba_clipped(canvas, resized, 0, 0)
                    resized = canvas
                    alpha_panel = resized[:, :, 3]
            glow_rgb = None
            if fg_glow > 0:
                glow = cv2.GaussianBlur(alpha_panel, (0, 0), fg_glow_blur)
                glow_rgb = np.dstack([glow, glow, glow]).astype(np.float32) * fg_glow
            resized = np.dstack([resized[:, :, :3], alpha_panel])
            mask_alpha = resized[:, :, 3]
            shadow = (
                cv2.GaussianBlur(mask_alpha, (0, 0), fg_shadow_blur)
                if fg_shadow_blur > 0
                else mask_alpha
            )

            def _pos(t):
                tt = t - fg_offset
                if tt <= dwell:
                    l, tp = left0, top0
                else:
                    p = (tt - dwell) / max(1e-6, travel)
                    p = ease_fn(p)
                    l = left0 + parallax_bg * (left1 - left0) * p
                    tp = top0 + parallax_bg * (top1 - top0) * p
                    if travel_path == "arc":
                        dx = left1 - left0
                        dy = top1 - top0
                        dist = math.hypot(dx, dy)
                        if dist > 1e-6:
                            px, py = -dy / dist, dx / dist
                            off = math.sin(math.pi * p) * dist * 0.25 * parallax_bg
                            l += px * off
                            tp += py * off
                nat_cx = (x + w / 2 - l) / win_w * Wout
                nat_cy = (y + h / 2 - tp) / win_h * Hout
                cam_dx = (l - left0) / win_w * Wout
                cam_dy = (tp - top0) / win_h * Hout
                dst_cx = nat_cx + parallax_fg * cam_dx
                dst_cy = nat_cy + parallax_fg * cam_dy
                x_pos = int(round(dst_cx - dst_w / 2))
                y_pos = int(round(dst_cy - dst_h / 2))
                return x_pos, y_pos

            def make_fg_frame(t, overlay=resized, shadow_map=shadow, glow_map=glow_rgb, seg_start=start, seg_idx=i):
                canvas = np.zeros((Hout, Wout, 4), dtype=np.uint8)
                scale = 1.0
                pop_dur = min(0.35, dwell)
                if overlay_pop < 1.0 and t < pop_dur:
                    p = 0.5 - 0.5 * math.cos(math.pi * t / pop_dur)
                    scale = overlay_pop + (1 - overlay_pop) * p
                if fg_drift_zoom > 0 and fg_drift_speed > 0:
                    phase = 2 * math.pi * fg_drift_speed * seg_start
                    scale *= 1.0 + fg_drift_zoom * math.sin(
                        phase + 2 * math.pi * fg_drift_speed * t
                    )
                x_pos, y_pos = _pos(t)
                use_overlay = overlay
                if scale != 1.0:
                    ow = max(1, int(round(dst_w * scale)))
                    oh = max(1, int(round(dst_h * scale)))
                    use_overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_CUBIC)
                    x_pos += (dst_w - ow) // 2
                    y_pos += (dst_h - oh) // 2
                if overlay_jitter > 0:
                    frame_idx = int(round(t * fps))
                    rng = np.random.default_rng(1337 + seg_idx * 10000 + frame_idx)
                    x_pos += int(round(rng.normal(0, overlay_jitter)))
                    y_pos += int(round(rng.normal(0, overlay_jitter)))
                if fg_shadow > 0:
                    sx = x_pos + fg_shadow_offset
                    sy = y_pos + fg_shadow_offset
                    _darken_region_with_alpha_clipped(
                        canvas[:, :, :3], shadow_map, sx, sy, float(fg_shadow)
                    )
                if glow_map is not None:
                    _add_rgb_clipped(canvas[:, :, :3], glow_map, x_pos, y_pos)
                _paste_rgba_clipped(canvas, use_overlay, x_pos, y_pos)
                return canvas[:, :, :3]

            def make_fg_mask(t, overlay=resized, seg_start=start, seg_idx=i):
                canvas = np.zeros((Hout, Wout, 4), dtype=np.uint8)
                scale = 1.0
                pop_dur = min(0.35, dwell)
                if overlay_pop < 1.0 and t < pop_dur:
                    p = 0.5 - 0.5 * math.cos(math.pi * t / pop_dur)
                    scale = overlay_pop + (1 - overlay_pop) * p
                if fg_drift_zoom > 0 and fg_drift_speed > 0:
                    phase = 2 * math.pi * fg_drift_speed * seg_start
                    scale *= 1.0 + fg_drift_zoom * math.sin(
                        phase + 2 * math.pi * fg_drift_speed * t
                    )
                x_pos, y_pos = _pos(t)
                use_overlay = overlay
                if scale != 1.0:
                    ow = max(1, int(round(dst_w * scale)))
                    oh = max(1, int(round(dst_h * scale)))
                    use_overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_CUBIC)
                    x_pos += (dst_w - ow) // 2
                    y_pos += (dst_h - oh) // 2
                if overlay_jitter > 0:
                    frame_idx = int(round(t * fps))
                    rng = np.random.default_rng(1337 + seg_idx * 10000 + frame_idx)
                    x_pos += int(round(rng.normal(0, overlay_jitter)))
                    y_pos += int(round(rng.normal(0, overlay_jitter)))
                _paste_rgba_clipped(canvas, use_overlay, x_pos, y_pos)
                return canvas[:, :, 3] / 255.0

        else:  # center mode
            ph, pw = panel.shape[:2]
            area_ratio = (ph * pw) / (Hpage * Wpage)
            ofit = overlay_fit
            if area_ratio < 0.08:
                ofit = min(ofit, 0.72)
            elif area_ratio > 0.25:
                ofit = min(ofit, 0.64)
            else:
                ofit = min(ofit, 0.66)
            scale = min(ofit * Hout / max(1, ph), Wout / max(1, pw))
            nw = int(round(pw * scale))
            nh = int(round(ph * scale))
            dx_max = abs(cx1 - cx0) * parallax_fg * (Wout / win_w)
            dy_max = abs(cy1 - cy0) * parallax_fg * (Hout / win_h)
            max_w = Wout - 2 * (overlay_margin + dx_max)
            max_h = Hout - 2 * (overlay_margin + dy_max)
            if nw > max_w or nh > max_h:
                scale = min(max_w / max(1, pw), max_h / max(1, ph))
                nw, nh = int(round(pw * scale)), int(round(ph * scale))
                logging.warning("overlay_fit reduced to avoid clipping")
            nw = min(nw, Wout - 2 * overlay_margin)
            nh = min(nh, Hout - 2 * overlay_margin)
            x_base = (Wout - nw) // 2
            y_base = (Hout - nh) // 2
            x_base = max(overlay_margin, min(x_base, Wout - overlay_margin - nw))
            y_base = max(overlay_margin, min(y_base, Hout - overlay_margin - nh))
            resized = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_CUBIC)
            alpha_panel = resized[:, :, 3]
            if overlay_edge == "feather":
                sigma = max(1, int(overlay_edge_strength * 10))
                alpha_panel = gaussian_blur(alpha_panel, sigma=sigma)
            elif overlay_edge == "torn":
                noise = np.random.rand(*alpha_panel.shape).astype(np.float32)
                noise = cv2.GaussianBlur(noise, (0, 0), 5)
                thr = 0.5 + overlay_edge_strength * 0.25
                alpha_panel = (alpha_panel.astype(np.float32) * (noise > thr)).astype(np.uint8)
            if overlay_frame_px > 0:
                k = np.ones((overlay_frame_px * 2 + 1, overlay_frame_px * 2 + 1), np.uint8)
                dil = cv2.dilate(alpha_panel, k, iterations=1)
                ring = cv2.subtract(dil, alpha_panel)
                if np.any(ring):
                    ring_rgba = np.zeros_like(resized)
                    ring_rgba[:, :, :3] = frame_rgb
                    ring_rgba[:, :, 3] = ring
                    canvas = np.zeros_like(resized)
                    _paste_rgba_clipped(canvas, ring_rgba, 0, 0)
                    _paste_rgba_clipped(canvas, resized, 0, 0)
                    resized = canvas
                    alpha_panel = resized[:, :, 3]
            glow_rgb = None
            if fg_glow > 0:
                glow = cv2.GaussianBlur(alpha_panel, (0, 0), fg_glow_blur)
                glow_rgb = np.dstack([glow, glow, glow]).astype(np.float32) * fg_glow
            resized = np.dstack([resized[:, :, :3], alpha_panel])
            mask_alpha = resized[:, :, 3]
            if fg_shadow_blur > 0:
                shadow = cv2.GaussianBlur(mask_alpha, (0, 0), fg_shadow_blur)
            else:
                shadow = mask_alpha
            scale_x = Wout / win_w
            scale_y = Hout / win_h

            def _pos(t, xb=x_base, yb=y_base):
                offx = offy = 0.0
                tt = t - fg_offset
                if tt > dwell:
                    p = (tt - dwell) / max(1e-6, travel)
                    p = ease_fn(p)
                    offx = (cx1 - cx0) * parallax_fg * p * scale_x
                    offy = (cy1 - cy0) * parallax_fg * p * scale_y
                    if travel_path == "arc":
                        dx = (cx1 - cx0) * scale_x
                        dy = (cy1 - cy0) * scale_y
                        dist = math.hypot(dx, dy)
                        if dist > 1e-6:
                            px, py = -dy / dist, dx / dist
                            off = math.sin(math.pi * p) * dist * 0.25 * parallax_fg
                            offx += px * off
                            offy += py * off
                x_pos = int(round(xb + offx))
                y_pos = int(round(yb + offy))
                x_pos = max(0, min(x_pos, Wout - nw))
                y_pos = max(0, min(y_pos, Hout - nh))
                return x_pos, y_pos

            def make_fg_frame(t, overlay=resized, shadow_map=shadow, glow_map=glow_rgb, seg_start=start, seg_idx=i):
                canvas = np.zeros((Hout, Wout, 4), dtype=np.uint8)
                scale = 1.0
                pop_dur = min(0.35, dwell)
                if overlay_pop < 1.0 and t < pop_dur:
                    p = 0.5 - 0.5 * math.cos(math.pi * t / pop_dur)
                    scale = overlay_pop + (1 - overlay_pop) * p
                if fg_drift_zoom > 0 and fg_drift_speed > 0:
                    phase = 2 * math.pi * fg_drift_speed * seg_start
                    scale *= 1.0 + fg_drift_zoom * math.sin(
                        phase + 2 * math.pi * fg_drift_speed * t
                    )
                x_pos, y_pos = _pos(t)
                use_overlay = overlay
                if scale != 1.0:
                    ow = max(1, int(round(nw * scale)))
                    oh = max(1, int(round(nh * scale)))
                    use_overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_CUBIC)
                    x_pos += (nw - ow) // 2
                    y_pos += (nh - oh) // 2
                if overlay_jitter > 0:
                    frame_idx = int(round(t * fps))
                    rng = np.random.default_rng(1337 + seg_idx * 10000 + frame_idx)
                    x_pos += int(round(rng.normal(0, overlay_jitter)))
                    y_pos += int(round(rng.normal(0, overlay_jitter)))
                if fg_shadow > 0:
                    sx = x_pos + fg_shadow_offset
                    sy = y_pos + fg_shadow_offset
                    _darken_region_with_alpha_clipped(
                        canvas[:, :, :3], shadow_map, sx, sy, float(fg_shadow)
                    )
                if glow_map is not None:
                    _add_rgb_clipped(canvas[:, :, :3], glow_map, x_pos, y_pos)
                _paste_rgba_clipped(canvas, use_overlay, x_pos, y_pos)
                return canvas[:, :, :3]

            def make_fg_mask(t, overlay=resized, seg_start=start, seg_idx=i):
                canvas = np.zeros((Hout, Wout, 4), dtype=np.uint8)
                scale = 1.0
                pop_dur = min(0.35, dwell)
                if overlay_pop < 1.0 and t < pop_dur:
                    p = 0.5 - 0.5 * math.cos(math.pi * t / pop_dur)
                    scale = overlay_pop + (1 - overlay_pop) * p
                if fg_drift_zoom > 0 and fg_drift_speed > 0:
                    phase = 2 * math.pi * fg_drift_speed * seg_start
                    scale *= 1.0 + fg_drift_zoom * math.sin(
                        phase + 2 * math.pi * fg_drift_speed * t
                    )
                x_pos, y_pos = _pos(t)
                use_overlay = overlay
                if scale != 1.0:
                    ow = max(1, int(round(nw * scale)))
                    oh = max(1, int(round(nh * scale)))
                    use_overlay = cv2.resize(overlay, (ow, oh), interpolation=cv2.INTER_CUBIC)
                    x_pos += (nw - ow) // 2
                    y_pos += (nh - oh) // 2
                if overlay_jitter > 0:
                    frame_idx = int(round(t * fps))
                    rng = np.random.default_rng(1337 + seg_idx * 10000 + frame_idx)
                    x_pos += int(round(rng.normal(0, overlay_jitter)))
                    y_pos += int(round(rng.normal(0, overlay_jitter)))
                _paste_rgba_clipped(canvas, use_overlay, x_pos, y_pos)
                return canvas[:, :, 3] / 255.0

        fg = _set_fps(VideoClip(make_fg_frame, duration=dwell + travel + settle), fps)
        mask_clip = VideoClip(make_fg_mask, duration=dwell + travel + settle)
        mask_clip.ismask = True
        fg_mask = _set_fps(mask_clip, fps)
        fg = _attach_mask(fg, fg_mask)
        fg_clips.append(fg)

        if debug_overlay and i < 2:
            t_dbg = min(dwell / 2, max(0.0, dwell - 1e-3))
            bg_dbg = make_bg(t_dbg)
            fg_dbg = np.zeros((Hout, Wout, 4), dtype=np.uint8)
            x_dbg, y_dbg = _pos(t_dbg)
            _paste_rgba_clipped(fg_dbg, resized, x_dbg, y_dbg)
            alpha = fg_dbg[:, :, 3:4] / 255.0
            comp_dbg = (fg_dbg[:, :, :3] * alpha + bg_dbg * (1 - alpha)).astype(np.uint8)
            for gx in range(0, Wout, 20):
                cv2.line(comp_dbg, (gx, 0), (gx, Hout - 1), (0, 255, 0), 1)
            for gy in range(0, Hout, 20):
                cv2.line(comp_dbg, (0, gy), (Wout - 1, gy), (0, 255, 0), 1)
            cv2.imwrite(f"debug_overlay_{i:04d}.png", cv2.cvtColor(comp_dbg, cv2.COLOR_RGB2BGR))

    n = len(fg_clips)
    seq: List[VideoClip] = []
    for i in range(n):
        s = seg_timings[i]
        dwell_i, travel_i, settle_i = s.dwell, s.travel, s.settle
        comp = _with_duration(
            CompositeVideoClip([bg_clips[i], fg_clips[i]], size=target_size),
            dwell_i + travel_i + settle_i,
        )
        comp = _set_fps(comp, fps)
        seg_dur = dwell_i + settle_i
        seq.append(comp.subclip(0, seg_dur))
        if i < n - 1:
            vec = (
                items[i + 1]["center"][0] - items[i]["center"][0],
                items[i + 1]["center"][1] - items[i]["center"][1],
            )
            tail_bg = bg_clips[i].subclip(0, seg_dur + trans_dur)
            tail_fg = fg_clips[i].subclip(0, seg_dur + trans_dur)
            if trans == "smear":
                tclip = smear_bg_crossfade_fg(
                    tail_bg,
                    bg_clips[i + 1],
                    tail_fg,
                    fg_clips[i + 1],
                    trans_dur,
                    target_size,
                    vec,
                    strength=smear_strength,
                    fps=fps,
                )
            elif trans == "whip":
                bg_t = whip_pan_transition(
                    tail_bg, bg_clips[i + 1], trans_dur, target_size, vec, fps=fps
                )
                fg_t = _with_duration(
                    CompositeVideoClip(
                        [
                            tail_fg.subclip(seg_dur, seg_dur + trans_dur)
                            .crossfadeout(trans_dur),
                            fg_clips[i + 1]
                            .subclip(0, trans_dur)
                            .crossfadein(trans_dur),
                        ],
                        size=target_size,
                    ),
                    trans_dur,
                )
                fg_t = _set_fps(fg_t, fps)
                tclip = _with_duration(
                    CompositeVideoClip([bg_t, fg_t], size=target_size),
                    trans_dur,
                )
                tclip = _set_fps(tclip, fps)
            else:
                prev_comp = CompositeVideoClip(
                    [bg_clips[i], fg_clips[i]], size=target_size
                )
                next_comp = CompositeVideoClip(
                    [bg_clips[i + 1], fg_clips[i + 1]], size=target_size
                )
                if trans == "slide":
                    tclip = slide_transition(
                        prev_comp, next_comp, trans_dur, target_size, fps
                    )
                else:
                    tail = prev_comp.subclip(seg_dur, seg_dur + trans_dur)
                    head = next_comp.subclip(0, trans_dur)
                    tclip = _with_duration(
                        CompositeVideoClip(
                            [tail.crossfadeout(trans_dur), head.crossfadein(trans_dur)],
                            size=target_size,
                        ),
                        trans_dur,
                    )
                    tclip = _set_fps(tclip, fps)
            seq.append(tclip)

    final = concatenate_videoclips(seq, method="compose")
    return _set_fps(final, fps)


def _export_profile(profile: str, codec: str, target_size: Tuple[int, int]) -> Dict[str, object]:
    """Return export settings for given *profile* and *codec*."""
    base = {
        "preview": {"crf": "31", "preset": "veryfast", "audio_bitrate": "96k", "fps": 24},
        "social": {"crf": "26", "preset": "medium", "audio_bitrate": "128k", "fps": 30},
        "quality": {"crf": "21", "preset": "slow", "audio_bitrate": "192k", "fps": 30},
    }[profile]
    resize = None
    if profile == "preview" and target_size == (1080, 1920):
        resize = (720, 1280)
    codec_map = {"h264": "libx264", "hevc": "libx265"}
    ffmpeg_params = ["-movflags", "+faststart", "-crf", base["crf"], "-pix_fmt", "yuv420p"]
    if codec == "hevc":
        ffmpeg_params.extend(["-tag:v", "hvc1"])
    return {
        "fps": base["fps"],
        "resize": resize,
        "ffmpeg_params": ffmpeg_params,
        "audio_bitrate": base["audio_bitrate"],
        "codec": codec_map[codec],
        "audio_codec": "aac",
        "preset": base["preset"],
    }


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
    return _set_fps(overlay_caption(zoomed, caption, screen_size), fps)


def make_filmstrip(
    input_folder: str,
    audio_fit: str = "trim",
    profile: str = "social",
    codec: str = "h264",
    target_size: ScreenSize = (1080, 1920),
) -> str:
    """Build final video from assets in *input_folder*.

    Parameters
    ----------
    input_folder:
        Directory with images and audio.
    audio_fit:
        Strategy for matching audio length.
    preview:
        If true, export using lightweight preview settings.
    """
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
        clip = ken_burns_scroll(path, target_size, duration, 30, focus_point, caption)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    video_duration = sum(c.duration for c in clips)
    audio = _fit_audio_clip(audio_path, video_duration, audio_fit)
    final_clip = final_clip.set_audio(audio)
    output_path = os.path.join(input_folder, "final_video.mp4")
    prof = _export_profile(profile, codec, target_size)
    if prof.get("resize"):
        final_clip = final_clip.resize(newsize=prof["resize"])
    final_clip.write_videofile(
        output_path,
        fps=prof["fps"],
        codec=prof["codec"],
        audio_codec=prof["audio_codec"],
        audio_bitrate=prof["audio_bitrate"],
        ffmpeg_params=prof["ffmpeg_params"],
        preset=prof["preset"],
    )
    return output_path
