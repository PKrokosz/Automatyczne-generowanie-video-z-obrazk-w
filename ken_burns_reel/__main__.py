"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import os

from .bin_config import resolve_imagemagick, resolve_tesseract
from .builder import make_filmstrip
from .ocr import verify_tesseract_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Ken Burns style video")
    parser.add_argument("folder", help="Input folder with images and audio")
    parser.add_argument("--tesseract", help="Path to Tesseract binary")
    parser.add_argument("--magick", help="Path to ImageMagick binary")
    parser.add_argument(
        "--mode",
        choices=["classic", "panels"],
        default="classic",
        help=(
            "classic: dotychczasowy montaż; panels: ruch kamery po panelach komiksu"
        ),
    )
    parser.add_argument("--dwell", type=float, default=1.0, help="Czas zatrzymania na panelu (s)")
    parser.add_argument("--travel", type=float, default=0.6, help="Czas przejazdu między panelami (s)")
    parser.add_argument("--xfade", type=float, default=0.4, help="Crossfade między stronami (s)")
    parser.add_argument("--settle", type=float, default=0.1, help="Długość micro-holdu (s)")
    parser.add_argument(
        "--easing",
        choices=["ease", "linear"],
        default="ease",
        help="Rodzaj easing przy przejazdach",
    )
    parser.add_argument(
        "--dwell-scale",
        type=float,
        default=1.0,
        help="Globalne skalowanie czasu zatrzymania po zważeniu",
    )
    parser.add_argument(
        "--align-beat",
        action="store_true",
        help="Wyrównaj start stron do najbliższego beatu",
    )
    parser.add_argument(
        "--debug-panels",
        action="store_true",
        help=(
            "Tryb debug – zapisuje plik panels_debug.jpg z wykrytymi ramkami i kończy działanie."
        ),
    )
    args = parser.parse_args()

    resolve_imagemagick(args.magick)
    resolve_tesseract(args.tesseract)
    verify_tesseract_available()

    if args.debug_panels:
        from .panels import debug_detect_panels

        debug_detect_panels(args.folder)
        print("✅ Zapisano panels_debug.jpg – sprawdź kolejność ramek.")
        return

    if args.mode == "panels":
        from .builder import make_panels_cam_sequence
        from .audio import extract_beats
        from moviepy.editor import AudioFileClip
        from moviepy.audio.fx import audio_fadein, audio_fadeout

        images = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        images.sort(key=lambda s: os.path.basename(s).lower())
        if not images:
            raise FileNotFoundError("Brak obrazów w folderze.")

        beat_times = None
        audios = [
            f
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".mp3", ".wav", ".m4a"}
        ]
        audio_path = None
        if audios:
            audio_path = os.path.join(args.folder, audios[0])
            if args.align_beat:
                beat_times = extract_beats(audio_path)

        clip = make_panels_cam_sequence(
            images,
            target_size=(1080, 1920),
            fps=30,
            dwell=args.dwell,
            travel=args.travel,
            xfade=args.xfade,
            settle=args.settle,
            easing=args.easing,
            dwell_scale=args.dwell_scale,
            align_beat=args.align_beat,
            beat_times=beat_times,
        )
        if audio_path:
            audioclip = AudioFileClip(audio_path)
            audioclip = audio_fadein.audio_fadein(audioclip, 0.15)
            audioclip = audio_fadeout.audio_fadeout(audioclip, 0.15)
            clip = clip.set_audio(audioclip)
        out_path = os.path.join(args.folder, "final_video.mp4")
        clip.write_videofile(out_path, fps=30, codec="libx264")
    else:
        make_filmstrip(args.folder)


if __name__ == "__main__":
    main()
