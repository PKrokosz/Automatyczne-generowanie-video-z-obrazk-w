"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import os

from .bin_config import resolve_imagemagick, resolve_tesseract
from .builder import make_filmstrip, _export_profile
from .ocr import verify_tesseract_available


def _page_scale_type(x: str) -> float:
    v = float(x)
    if not (0.80 < v <= 1.0):
        raise argparse.ArgumentTypeError("--page-scale must be in (0.80,1.0]")
    return v


def _parallax_type(x: str) -> float:
    v = float(x)
    if not (0.0 <= v <= 1.0):
        raise argparse.ArgumentTypeError("--bg-parallax must be in [0,1]")
    return v


def _nonneg_int(x: str) -> int:
    v = int(x)
    if v < 0:
        raise argparse.ArgumentTypeError("--panel-bleed must be >= 0")
    return v


def _zoom_max_type(x: str) -> float:
    v = float(x)
    if v < 1.0:
        raise argparse.ArgumentTypeError("--zoom-max must be >= 1.0")
    return v

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
    parser.add_argument("--settle", type=float, default=0.14, help="Długość micro-holdu (s)")
    parser.add_argument(
        "--travel-ease",
        "--easing",
        dest="travel_ease",
        choices=["in", "out", "inout", "linear", "ease"],
        default="inout",
        help="Profil jazdy kamery",
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
    parser.add_argument(
        "--audio-fit",
        choices=["trim", "silence", "loop"],
        default="trim",
        help="Jak dopasować audio do długości wideo",
    )
    parser.add_argument(
        "--dwell-mode",
        choices=["first", "each"],
        default="first",
        help="Na ilu panelach zatrzymywać się w pełni",
    )
    parser.add_argument(
        "--bg-mode",
        choices=["none", "blur", "stretch", "gradient"],
        default="blur",
        help="Underlay pod stroną",
    )
    parser.add_argument(
        "--page-scale",
        type=_page_scale_type,
        default=0.92,
        help="Skala foreground (mniejsza niż 1.0 = widać tło)",
    )
    parser.add_argument(
        "--bg-parallax",
        type=_parallax_type,
        default=0.85,
        help="Siła paralaksy tła podczas travelu",
    )
    parser.add_argument(
        "--panel-bleed",
        type=_nonneg_int,
        default=24,
        help="Margines przy kadrowaniu panelu (px)",
    )
    parser.add_argument(
        "--zoom-max",
        type=_zoom_max_type,
        default=1.06,
        help="Maksymalne dodatkowe przybliżenie dla małego tekstu",
    )
    parser.add_argument(
        "--profile",
        choices=["preview", "social", "quality"],
        default="social",
        help="Preset eksportu",
    )
    parser.add_argument("--preview", action="store_true", help="Skrót dla --profile preview")
    parser.add_argument(
        "--codec",
        choices=["h264", "hevc"],
        default="h264",
        help="Kodek wideo",
    )
    parser.add_argument("--size", help="Docelowy rozmiar WxH")
    parser.add_argument(
        "--aspect",
        choices=["9:16", "16:9", "1:1"],
        help="Proporcje (z --height)",
    )
    parser.add_argument("--height", type=int, help="Wysokość dla --aspect")
    args = parser.parse_args()

    if args.preview:
        args.profile = "preview"

    if args.travel_ease == "ease":
        args.travel_ease = "inout"

    target_size = (1080, 1920)
    if args.size:
        try:
            w, h = args.size.lower().split("x")
            target_size = (int(w), int(h))
        except Exception as e:  # pragma: no cover - argparse ensures format
            raise argparse.ArgumentTypeError("--size format WxH") from e
    elif args.aspect and args.height:
        ratios = {"9:16": 9 / 16, "16:9": 16 / 9, "1:1": 1.0}
        ratio = ratios[args.aspect]
        h = args.height
        w = int(round(h * ratio))
        target_size = (w, h)

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
            target_size=target_size,
            fps=30,
            dwell=args.dwell,
            travel=args.travel,
            xfade=args.xfade,
            settle=args.settle,
            travel_ease=args.travel_ease,
            dwell_scale=args.dwell_scale,
            align_beat=args.align_beat,
            beat_times=beat_times,
            audio_path=audio_path,
            audio_fit=args.audio_fit,
            dwell_mode=args.dwell_mode,
            bg_mode=args.bg_mode,
            page_scale=args.page_scale,
            bg_parallax=args.bg_parallax,
            panel_bleed=args.panel_bleed,
            zoom_max=args.zoom_max,
        )
        out_path = os.path.join(args.folder, "final_video.mp4")
        prof = _export_profile(args.profile, args.codec, target_size)
        if prof.get("resize"):
            clip = clip.resize(newsize=prof["resize"])
        clip.write_videofile(
            out_path,
            fps=prof["fps"],
            codec=prof["codec"],
            audio_codec=prof["audio_codec"],
            audio_bitrate=prof["audio_bitrate"],
            ffmpeg_params=prof["ffmpeg_params"],
            preset=prof["preset"],
        )
    else:
        make_filmstrip(
            args.folder,
            audio_fit=args.audio_fit,
            profile=args.profile,
            codec=args.codec,
            target_size=target_size,
        )


if __name__ == "__main__":
    main()
