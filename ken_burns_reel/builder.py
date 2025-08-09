"""Core building logic for the Ken Burns reel."""
from __future__ import annotations

import glob
import math
import os
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


def _get_ease_fn(name: str):
    return {
        "linear": lambda t: t,
        "in": ease_in,
        "out": ease_out,
        "inout": ease_in_out,
    }.get(name, ease_in_out)


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
    return final


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
    overlay_fit: float = 0.75,
    overlay_margin: int = 0,
    bg_source: str = "page",
    bg_tone_strength: float = 0.7,
    parallax_bg: float = 0.85,
    parallax_fg: float = 0.0,
    fg_shadow: float = 0.25,
    fg_shadow_blur: int = 18,
    fg_shadow_offset: int = 4,
    fg_shadow_mode: str = "soft",
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
            boxes = order_panels_lr_tb(detect_panels(im))
        panel_folder = os.path.join(panels_dir, f"page_{idx:04d}")
        panel_files = sorted(glob.glob(os.path.join(panel_folder, "panel_*.png")))
        for box, panel_file in zip(boxes, panel_files):
            items.append({"page": page_arr, "panel": panel_file, "box": box})
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

    bg_clips: List[VideoClip] = []
    fg_clips: List[VideoClip] = []

    ease_fn = _get_ease_fn(travel_ease)
    parallax_fg = max(0.0, min(0.5, parallax_fg))
    bg_tone_strength = max(0.0, min(1.0, bg_tone_strength))

    if fg_shadow_mode == "hard":
        fg_shadow_blur = max(1, fg_shadow_blur // 2)
        fg_shadow += 0.05
    fg_shadow = max(0.0, min(0.5, fg_shadow))

    for i, it in enumerate(items):
        page_arr = it["page"]
        panel_arr = it["panel_arr"]
        box = it["box"]
        center = it["center"]
        end_center = center
        end_box = box
        if i + 1 < len(items) and items[i + 1]["page"] is page_arr:
            end_center = items[i + 1]["center"]
            end_box = items[i + 1]["box"]
        Hpage, Wpage = page_arr.shape[:2]
        cx0, cy0, w0, h0 = _fit_window_to_box(Wpage, Hpage, box, target_size)
        cx1, cy1, w1, h1 = _fit_window_to_box(Wpage, Hpage, end_box, target_size)
        win_w = max(w0, w1)
        win_h = max(h0, h1)
        left0 = int(max(0, min(cx0 - win_w // 2, Wpage - win_w)))
        top0 = int(max(0, min(cy0 - win_h // 2, Hpage - win_h)))
        left1 = int(max(0, min(cx1 - win_w // 2, Wpage - win_w)))
        top1 = int(max(0, min(cy1 - win_h // 2, Hpage - win_h)))

        def make_bg(t, arr=page_arr, l0=left0, t0=top0, l1=left1, t1=top1):
            if bg_source == "page":
                if t <= dwell:
                    l, tp = l0, t0
                else:
                    p = (t - dwell) / max(1e-6, travel)
                    p = ease_fn(p)
                    l = l0 + parallax_bg * (l1 - l0) * p
                    tp = t0 + parallax_bg * (t1 - t0) * p
                l = int(max(0, min(l, Wpage - win_w)))
                tp = int(max(0, min(tp, Hpage - win_h)))
                crop = arr[tp : tp + win_h, l : l + win_w]
                frame_raw = cv2.resize(crop, (Wout, Hout), interpolation=cv2.INTER_CUBIC)
                if bg_tone_strength > 0:
                    hsv = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 1] *= 0.8
                    hsv[:, :, 2] *= 0.85
                    toned = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                    yy, xx = np.mgrid[0:Hout, 0:Wout]
                    dist = np.sqrt((xx - Wout / 2) ** 2 + (yy - Hout / 2) ** 2)
                    dist /= dist.max() + 1e-6
                    vign = 1 - 0.15 * dist
                    toned = (toned.astype(np.float32) * vign[..., None]).astype(np.uint8)
                    frame = cv2.addWeighted(
                        frame_raw.astype(np.float32),
                        1 - bg_tone_strength,
                        toned.astype(np.float32),
                        bg_tone_strength,
                        0,
                    ).astype(np.uint8)
                else:
                    frame = frame_raw
            else:
                frame = _make_underlay(arr, target_size, bg_source)
            return frame

        bg = _set_fps(VideoClip(make_bg, duration=dwell + travel), fps)
        bg = bg.set_mask(ImageClip(np.zeros((Hout, Wout)), ismask=True).set_duration(dwell + travel))
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
        ph, pw = panel.shape[:2]
        scale = min(overlay_fit * Hout / max(1, ph), Wout / max(1, pw))
        nw = int(round(pw * scale))
        nh = int(round(ph * scale))
        nw = min(nw, Wout - 2 * overlay_margin)
        nh = min(nh, Hout - 2 * overlay_margin)
        x_base = (Wout - nw) // 2
        y_base = (Hout - nh) // 2
        x_base = max(overlay_margin, min(x_base, Wout - overlay_margin - nw))
        y_base = max(overlay_margin, min(y_base, Hout - overlay_margin - nh))
        resized = cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_CUBIC)
        mask_alpha = resized[:, :, 3]
        alpha_norm = mask_alpha.astype(np.float32) / 255.0
        rgb = resized[:, :, :3]
        scale_x = Wout / win_w
        scale_y = Hout / win_h

        def _pos(t, xb=x_base, yb=y_base):
            offx = offy = 0.0
            if t > dwell:
                p = (t - dwell) / max(1e-6, travel)
                p = ease_fn(p)
                offx = (cx1 - cx0) * parallax_fg * p * scale_x
                offy = (cy1 - cy0) * parallax_fg * p * scale_y
            x_pos = int(round(xb + offx))
            y_pos = int(round(yb + offy))
            x_pos = max(0, min(x_pos, Wout - nw))
            y_pos = max(0, min(y_pos, Hout - nh))
            return x_pos, y_pos

        def make_fg_frame(t, rgb=rgb, alpha=mask_alpha):
            canvas = np.zeros((Hout, Wout, 3), dtype=np.uint8)
            x_pos, y_pos = _pos(t)
            if fg_shadow > 0:
                if fg_shadow_blur > 0:
                    shadow = gaussian_blur(alpha, sigma=fg_shadow_blur)
                else:
                    shadow = alpha
                shadow = shadow.astype(np.float32) * (fg_shadow / 255.0)
                sx = x_pos + fg_shadow_offset
                sy = y_pos + fg_shadow_offset
                ex = min(Wout, sx + nw)
                ey = min(Hout, sy + nh)
                sh = shadow[: ey - sy, : ex - sx][..., None]
                region = canvas[sy:ey, sx:ex].astype(np.float32)
                canvas[sy:ey, sx:ex] = np.clip(region * (1 - sh), 0, 255)
            ex = x_pos + nw
            ey = y_pos + nh
            canvas[y_pos:ey, x_pos:ex] = rgb
            return canvas

        def make_fg_mask(t, alpha=alpha_norm):
            mask = np.zeros((Hout, Wout), dtype=np.float32)
            x_pos, y_pos = _pos(t)
            ex = x_pos + nw
            ey = y_pos + nh
            mask[y_pos:ey, x_pos:ex] = alpha
            return mask

        fg = _set_fps(VideoClip(make_fg_frame, duration=dwell + travel), fps)
        fg_mask = _set_fps(
            VideoClip(make_fg_mask, duration=dwell + travel, ismask=True), fps
        )
        fg = fg.set_mask(fg_mask)
        fg_clips.append(fg)

    seq: List[VideoClip] = []
    for i in range(len(fg_clips)):
        comp = (
            _set_fps(
                CompositeVideoClip([bg_clips[i], fg_clips[i]], size=target_size)
                .set_duration(dwell + travel),
                fps,
            )
        )
        seq.append(comp.subclip(0, dwell))
        if i < len(fg_clips) - 1:
            vec = (
                items[i + 1]["center"][0] - items[i]["center"][0],
                items[i + 1]["center"][1] - items[i]["center"][1],
            )
            if trans == "smear":
                tclip = smear_bg_crossfade_fg(
                    bg_clips[i],
                    bg_clips[i + 1],
                    fg_clips[i],
                    fg_clips[i + 1],
                    trans_dur,
                    target_size,
                    vec,
                    strength=smear_strength,
                    fps=fps,
                )
            elif trans == "whip":
                bg_t = whip_pan_transition(
                    bg_clips[i], bg_clips[i + 1], trans_dur, target_size, vec, fps=fps
                )
                fg_t = _set_fps(
                    CompositeVideoClip(
                        [
                            fg_clips[i]
                            .subclip(dwell, dwell + trans_dur)
                            .crossfadeout(trans_dur),
                            fg_clips[i + 1]
                            .subclip(0, trans_dur)
                            .crossfadein(trans_dur),
                        ],
                        size=target_size,
                    ).set_duration(trans_dur),
                    fps,
                )
                tclip = _set_fps(
                    CompositeVideoClip([bg_t, fg_t], size=target_size).set_duration(trans_dur),
                    fps,
                )
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
                    tail = prev_comp.subclip(dwell, dwell + trans_dur)
                    head = next_comp.subclip(0, trans_dur)
                    tclip = _set_fps(
                        CompositeVideoClip(
                            [tail.crossfadeout(trans_dur), head.crossfadein(trans_dur)],
                            size=target_size,
                        ).set_duration(trans_dur),
                        fps,
                    )
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
    return overlay_caption(zoomed, caption, screen_size)


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
