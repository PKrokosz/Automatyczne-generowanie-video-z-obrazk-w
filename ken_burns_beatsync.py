# ken_burns_scroll_audio.py (rozszerzona wersja z komentarzami i możliwościami rozbudowy)

import os
import re
import numpy as np
import librosa
from moviepy.editor import (
    ImageClip, CompositeVideoClip, concatenate_videoclips, vfx, AudioFileClip
)
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}

# 📂 Zbiera ścieżki do obrazów w folderze

def list_images(folder):
    return sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if os.path.splitext(f.lower())[1] in IMAGE_EXTS],
        key=lambda f: [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', f)]
    )

# 🎵 Szuka pliku audio w folderze

def find_audio(folder):
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in AUDIO_EXTS:
            return os.path.join(folder, f)
    return None

# 🥁 Wydobywa beaty z pliku audio (librosa)

def extract_beats(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr).tolist()

# ✂️ Kadrowanie obrazu do formatu pionowego (z marginesem zoomu)

def smart_crop(img, target_w, target_h):
    w, h = img.size
    src_ratio = w / h
    target_ratio = target_w / target_h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

# 📈 Krzywa ease-in-out

def ease(t): return 3*t**2 - 2*t**3

# 🔄 Główna funkcja efektu Ken Burns z dryfem i zoomem

def ken_burns_scroll(img_path, size, duration, fps, timing=None):
    W, H = size
    img = Image.open(img_path).convert("RGB")
    img = smart_crop(img, int(W * 1.3), int(H * 1.3))  # powiększamy kadr by móc zoomować
    np_img = np.array(img)
    base_clip = ImageClip(np_img)

    start_zoom = 1.3
    end_zoom = 1.0
    start_x = int(W * 0.3)
    start_y = int(H * 0.3)

    if timing:
        duration = timing[1] - timing[0]

    def dynamic_position(t):
        p = t / duration
        dx = -ease(p) * W * 0.3  # przesunięcie w lewo
        dy = ease(p) * H * 0.4   # opadanie w dół
        return (start_x + dx, start_y + dy)

    def zoom_func(t):
        z = start_zoom + (end_zoom - start_zoom) * ease(t / duration)
        return z

    zoomed_clip = base_clip.resize(lambda t: zoom_func(t)).set_position(dynamic_position)
    return zoomed_clip.set_duration(duration).set_fps(fps)

# ➡️ Płynne przejście między panelami (slajd)

def slide_transition(prev_clip, next_clip, duration, size, fps):
    W, H = size
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)

    def move_left(t): return (-ease(t / duration) * W, 0)
    def move_right(t): return ((1 - ease(t / duration)) * W, 0)

    return CompositeVideoClip([
        tail.set_pos(move_left),
        head.set_pos(move_right)
    ], size=size).set_duration(duration).set_fps(fps)

# 🎬 Główna funkcja tworzenia filmiku

def make_filmstrip(folder, output="reel_filmstrip.mp4"):
    size = (1080, 1920)  # format pionowy
    transition = 0.5
    fps = 30

    images = list_images(folder)
    if not images:
        raise ValueError("Brak obrazów w folderze.")

    audio_path = find_audio(folder)
    if audio_path:
        beat_times = extract_beats(audio_path)
        if len(beat_times) < len(images) + 1:
            beat_times = np.linspace(0, len(images) * 3, len(images) + 1)
    else:
        beat_times = np.linspace(0, len(images) * 3, len(images) + 1)

    clips = []
    for i, path in enumerate(images):
        t0, t1 = beat_times[i], beat_times[i+1]
        clip = ken_burns_scroll(path, size, t1 - t0, fps, (t0, t1))
        clips.append(clip)
        if i < len(images) - 1:
            next_clip = ken_burns_scroll(images[i+1], size, beat_times[i+2] - beat_times[i+1], fps, (beat_times[i+1], beat_times[i+2]))
            clips.append(slide_transition(clip, next_clip, transition, size, fps))

    final = concatenate_videoclips(clips, method="compose")

    if audio_path:
        audio = AudioFileClip(audio_path).set_duration(final.duration)
        final = final.set_audio(audio)

    # 🧩 Eksport z bezpiecznym marginesem na overlay TikTok (dół + góra)
    safe_margin = 250  # px
    final = final.margin(top=safe_margin, bottom=safe_margin, opacity=0)

    final.write_videofile(
        output,
        fps=fps,
        codec="libx264",
        audio=bool(audio_path),
        preset="medium",
        threads=4,
        ffmpeg_params=["-crf", "22"]
    )

# ✅ PROPOZYCJE ROZBUDOWY:
# 1. 📍 Wykrycie twarzy / tekstu do kadrowania startowego (np. OpenCV haarcascade / pytesseract).
# 2. ⚡ Preloading obrazów jako `ImageClip(...).fx(...)` w RAM przed montażem (dla szybkości).
# 3. 📐 Zmienne tempo Ken Burns (np. przyspieszenie na refrenie na podstawie beat intensity).
# 4. 🎚️ Interfejs sterowania dla ustawień zoomu, trajektorii, opóźnień itp. (np. .json config).
# 5. 🧪 Preview generator – tworzy GIF lub szybki podgląd .webm dla testów ruchu przed eksportem.

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Użycie: python ken_burns_reels.py folder_z_obrazami")
        sys.exit(1)
    make_filmstrip(sys.argv[1])
