"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import os

import moviepy.config as mpyconf
import pytesseract

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
    parser.add_argument(
        "--debug-panels",
        action="store_true",
        help=(
            "Tryb debug – zapisuje plik panels_debug.jpg z wykrytymi ramkami i kończy działanie."
        ),
    )
    args = parser.parse_args()

    if args.magick:
        os.environ["IMAGEMAGICK_BINARY"] = args.magick
        mpyconf.change_settings({"IMAGEMAGICK_BINARY": args.magick})
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    verify_tesseract_available()

    if args.debug_panels:
        from .panels import debug_detect_panels

        debug_detect_panels(args.folder)
        print("✅ Zapisano panels_debug.jpg – sprawdź kolejność ramek.")
        return

    if args.mode == "panels":
        from .builder import make_panels_cam_clip
        from moviepy.editor import AudioFileClip

        # wybieramy pierwsze zdjęcie z folderu
        images = [
            f
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not images:
            raise FileNotFoundError("Brak obrazów w folderze.")
        image_path = os.path.join(args.folder, images[0])
        clip = make_panels_cam_clip(image_path, target_size=(1080, 1920))

        # audio (opcjonalnie)
        audios = [
            f
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".mp3", ".wav", ".m4a"}
        ]
        if audios:
            clip = clip.set_audio(
                AudioFileClip(os.path.join(args.folder, audios[0]))
            )
        out_path = os.path.join(args.folder, "final_video.mp4")
        clip.write_videofile(out_path, fps=30, codec="libx264")
    else:
        make_filmstrip(args.folder)


if __name__ == "__main__":
    main()
