# ken_burns_scroll_audio.py – FIXED VERSION 🛠️

import os
import sys
import glob
import numpy as np
from moviepy.editor import (
    AudioFileClip, concatenate_videoclips, CompositeVideoClip,
    TextClip, ImageClip
)
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from PIL import Image
import cv2
import pytesseract
import librosa

# 🔧 Wymuszamy ścieżkę do ImageMagick (Windows fix)
os.environ['IMAGEMAGICK_BINARY'] = r"C:\\Program Files\\ImageMagick-7.1.2-Q16-HDRI\\magick.exe"
change_settings({"IMAGEMAGICK_BINARY": os.environ['IMAGEMAGICK_BINARY']})

# 🔧 Wymuszamy lokalizację tesseract (zmień jeśli inna)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# 📍 Sanityzacja OCR

def clean_text(text):
    return text.replace("\\", "").replace("\n", " ").strip()

# 🎯 OCR + Focus Point

def extract_caption_and_focus(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray)
    caption = pytesseract.image_to_string(image)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        focus_point = (int(x + w/2), int(y + h/2))
    else:
        focus_point = (image.shape[1]//2, image.shape[0]//2)
    return clean_text(caption), focus_point

# 🎬 Główna funkcja efektu

def ken_burns_scroll(image_path, screen_size, duration, fps, time_range, focus_point, caption):
    img = Image.open(image_path)
    img_clip = ImageClip(image_path).set_duration(duration)
    zoomed = img_clip.fx(crop, width=screen_size[0], height=screen_size[1],
                         x_center=focus_point[0], y_center=focus_point[1])
    return overlay_caption(zoomed, caption, screen_size)

# ✏️ Nakładanie tekstu z fallbackiem

def overlay_caption(clip, text, size):
    try:
        txt = TextClip(text, fontsize=50, color='white', stroke_color='black', stroke_width=2,
                       method='caption', size=(size[0] - 100, None))
    except Exception as e:
        print("⚠️ Fallback to method='label' due to:", e)
        txt = TextClip(text, fontsize=50, color='white', stroke_color='black', stroke_width=2,
                       method='label')
    txt = txt.set_position(('center', 'bottom')).set_duration(clip.duration)
    return CompositeVideoClip([clip, txt])

# 📼 Główna funkcja budowy filmu

def make_filmstrip(input_folder):
    print(f"🎬 Start montażu filmu z folderu: {input_folder}")
    image_files = sorted([f for f in glob.glob(os.path.join(input_folder, '*')) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"📸 Liczba obrazów: {len(image_files)}")

    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp3', '.wav'))]
    if not audio_files:
        raise FileNotFoundError("Brak pliku audio")
    audio_path = os.path.join(input_folder, audio_files[0])
    print(f"🎶 Znaleziono plik audio: {os.path.basename(audio_path)}")

    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"📈 Wykryto {len(beat_times)} beatów")

    clips = []
    for i, path in enumerate(image_files):
        print(f"📝 OCR dla: {path}")
        caption, focus_point = extract_caption_and_focus(path)
        t0 = beat_times[i] if i < len(beat_times) else beat_times[-1]
        t1 = beat_times[i + 1] if i + 1 < len(beat_times) else t0 + 0.6
        duration = t1 - t0
        print(f"⏱️ Czas klipu: {duration:.2f}s, Caption: {caption[:30]}...")
        clip = ken_burns_scroll(path, (1080, 1920), duration, 30, (t0, t1), focus_point, caption)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip = final_clip.set_audio(AudioFileClip(audio_path))
    output_path = os.path.join(input_folder, "final_video.mp4")
    final_clip.write_videofile(output_path, fps=30, codec='libx264')
    print(f"✅ Gotowe: {output_path}")

# ▶️ Uruchomienie
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Użycie: python ken_burns_scroll_audio.py <folder>")
        sys.exit(1)
    make_filmstrip(sys.argv[1])



# ✅ Konfiguracja ścieżki do ImageMagick (jeśli zainstalowane lokalnie)
import moviepy.config as mpyconf
mpyconf.change_settings({"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"})

# ✅ Konfiguracja ścieżki do Tesseract OCR
from shutil import which
pytesseract.pytesseract.tesseract_cmd = which("tesseract") or r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


def overlay_caption(clip: ImageClip, text: str, size: Tuple[int, int]) -> ImageClip:
    print(f"💬 Dodawanie captionu: {text}")
    if not text:
        return clip
    W, H = size
    txt = TextClip(text, fontsize=50, color='white', stroke_color='black', stroke_width=2, method='caption', size=(W - 100, None))
    txt = txt.set_position(("center", H - 200)).set_duration(clip.duration).fadein(0.5).fadeout(0.5)
    return CompositeVideoClip([clip, txt], size=size)

def extract_caption(img_path: str) -> str:
    print(f"📝 OCR dla: {img_path}")
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

def detect_focus_point(img: Image.Image) -> Tuple[int, int]:
    print("🔍 Szukam punktu skupienia (focus point)")
    gray = np.array(img.convert("L"))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        print("👤 Twarz znaleziona")
        return (x + w // 2, y + h // 2)
    else:
        print("⚠️ Brak twarzy, fallback do heurystyki jasności")
        brightness = np.array(img.convert("L"))
        y_indices, x_indices = np.indices(brightness.shape)
        total = brightness.sum()
        x = int((x_indices * brightness).sum() / total)
        y = int((y_indices * brightness).sum() / total)
        return (x, y)

def smart_crop(img, target_w, target_h):
    w, h = img.size
    src_ratio = w / h
    target_ratio = target_w / target_h
    print(f"✂️ Kadrowanie obrazu: ({w}x{h}) do ({target_w}x{target_h})")
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

def slide_transition(prev_clip, next_clip, duration, size, fps):
    W, H = size
    print(f"➡️ Tworzenie przejścia między klipami (czas: {duration}s)")
    tail = prev_clip.subclip(prev_clip.duration - duration, prev_clip.duration)
    head = next_clip.subclip(0, duration)

    def move_left(t): return (-ease(t / duration) * W, 0)
    def move_right(t): return ((1 - ease(t / duration)) * W, 0)

    return CompositeVideoClip([
        tail.set_pos(move_left),
        head.set_pos(move_right)
    ], size=size).set_duration(duration).set_fps(fps)

def extract_beats(audio_path):
    print(f"🥁 Analiza beatów: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    print(f"📈 Wykryto {len(beat_times)} beatów")
    return beat_times.tolist()

def verify_tesseract_available():
    binary = pytesseract.pytesseract.tesseract_cmd
    if not os.path.isfile(binary) and not which(binary):
        print(f"❌ Błąd: Nie znaleziono Tesseract-OCR pod: {binary}")
        print("   ➔ Upewnij się, że Tesseract jest zainstalowany:")
        print("   🔗 https://github.com/UB-Mannheim/tesseract/wiki")
        print("   ➔ Lub ustaw prawidłową ścieżkę w pytesseract.pytesseract.tesseract_cmd")
        sys.exit(1)
    else:
        print(f"✅ Wykryto Tesseract OCR: {binary}")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a"}
