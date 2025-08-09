"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import os

from .bin_config import resolve_imagemagick, resolve_tesseract
from .builder import make_filmstrip, _export_profile, _fit_audio_clip
from .ocr import verify_tesseract_available


def _page_scale_type(x: str) -> float:
    v = float(x)
    if not (0.80 < v <= 1.0):
        raise argparse.ArgumentTypeError("--page-scale must be in (0.80,1.0]")
    return v


def _parallax_type(x: str) -> float:
    v = float(x)
    return max(0.0, min(1.0, v))


def _parallax_fg_type(x: str) -> float:
    v = float(x)
    return max(0.0, min(0.5, v))


def _nonneg_int(x: str) -> int:
    v = int(x)
    if v < 0:
        raise argparse.ArgumentTypeError("--panel-bleed must be >= 0")
    return v


def _clamp_nonneg_int(x: str) -> int:
    return max(0, int(x))


def _zoom_max_type(x: str) -> float:
    v = float(x)
    if v < 1.0:
        raise argparse.ArgumentTypeError("--zoom-max must be >= 1.0")
    return v


def _run_oneclick(args: argparse.Namespace, target_size: tuple[int, int]) -> None:
    """Run simplified one-click workflow for comic pages."""

    from .panels import export_panels
    from .builder import make_panels_overlay_sequence
    from .audio import extract_beats
    import tempfile

    resolve_imagemagick(args.magick)
    resolve_tesseract(args.tesseract)
    verify_tesseract_available()

    pages_dir = os.path.join(args.folder, "pages")
    if not os.path.isdir(pages_dir):
        pages_dir = args.folder

    page_paths = [
        os.path.join(pages_dir, f)
        for f in os.listdir(pages_dir)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    page_paths.sort(key=lambda s: os.path.basename(s).lower())
    if not page_paths:
        raise FileNotFoundError("Brak obrazów stron.")

    with tempfile.TemporaryDirectory(prefix="panels_tmp") as tmpdir:
        for i, path in enumerate(page_paths, 1):
            out_sub = os.path.join(tmpdir, f"page_{i:04d}")
            export_panels(
                path,
                out_sub,
                mode="mask",
                bleed=12,
                tight_border=2,
                feather=2,
                gutter_thicken=args.gutter_thicken,
                min_area_ratio=args.min_panel_area_ratio,
            )

        beat_times = None
        audio_path = None
        audio_exts = {".mp3", ".wav", ".m4a"}
        candidates = []
        for base in {args.folder, os.path.dirname(args.folder)}:
            if os.path.isdir(base):
                for f in os.listdir(base):
                    if os.path.splitext(f)[1].lower() in audio_exts:
                        candidates.append(os.path.join(base, f))
        candidates.sort(key=lambda s: os.path.basename(s).lower())
        if candidates:
            audio_path = candidates[0]
            if args.align_beat:
                beat_times = extract_beats(audio_path)
        elif args.align_beat:
            print("⚠️ Nie znaleziono pliku audio – wideo bez wyrównania do beatów.")
            args.align_beat = False

        clip = make_panels_overlay_sequence(
            page_paths,
            tmpdir,
            target_size=target_size,
            fps=30,
            dwell=args.dwell,
            travel=args.travel,
            travel_ease="inout",
            align_beat=args.align_beat,
            beat_times=beat_times,
            overlay_fit=0.75,
            overlay_mode=args.overlay_mode,
            overlay_scale=args.overlay_scale,
            bg_source="page",
            parallax_bg=0.85,
            parallax_fg=0.08,
            min_panel_area_ratio=args.min_panel_area_ratio,
            gutter_thicken=args.gutter_thicken,
            debug_overlay=args.debug_overlay,
            limit_items=args.limit_items,
            trans="smear",
            trans_dur=0.30,
            smear_strength=1.1,
        )

        if audio_path:
            audio = _fit_audio_clip(audio_path, clip.duration, args.audio_fit)
            clip = clip.set_audio(audio)

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Ken Burns style video")
    parser.add_argument("folder", help="Input folder with images and audio")
    parser.add_argument("--tesseract", help="Path to Tesseract binary")
    parser.add_argument("--magick", help="Path to ImageMagick binary")
    parser.add_argument("--export-panels", help="Export detected panels to folder")
    parser.add_argument("--oneclick", action="store_true", help="Tryb one-click: auto video from pages and audio")
    parser.add_argument(
        "--export-mode",
        choices=["rect", "mask"],
        default="rect",
        help="Panel export mode",
    )
    parser.add_argument(
        "--mode",
        choices=["classic", "panels", "panels-items", "panels-overlay"],
        default=None,
        help=(
            "classic: dotychczasowy montaż; panels: ruch kamery po panelach komiksu; panels-items: montaż z pojedynczych paneli; panels-overlay: tło strona, foreground panel"
        ),
    )
    parser.add_argument("--limit-items", type=int, default=999, help="Limit liczby paneli w overlay")
    parser.add_argument("--tight-border", type=int, default=1, help="Erozja konturu w eksporcie mask (px)")
    parser.add_argument("--feather", type=int, default=1, help="Feather alpha w eksporcie mask (px)")
    parser.add_argument(
        "--enhance",
        choices=["none", "comic"],
        default="comic",
        help="Tryb poprawy paneli",
    )
    parser.add_argument("--enhance-strength", type=float, default=1.0, help="Siła enhance")
    parser.add_argument("--shadow", type=_parallax_type, default=0.2, help="Opacity cienia pod panelem")
    parser.add_argument("--shadow-blur", type=_clamp_nonneg_int, default=12, help="Rozmycie cienia (px)")
    parser.add_argument("--shadow-offset", type=_clamp_nonneg_int, default=3, help="Offset cienia (px)")
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
        "--trans",
        choices=["xfade", "slide", "smear", "whip"],
        default="smear",
        help="Przejście między panelami w trybie panels-items",
    )
    parser.add_argument(
        "--trans-dur",
        type=float,
        default=0.3,
        help="Długość przejścia między panelami (s)",
    )
    parser.add_argument(
        "--smear-strength",
        type=float,
        default=1.0,
        help="Siła smuga dla przejścia smear",
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

    def _overlay_fit_type(x: str) -> float:
        v = float(x)
        return max(0.0, min(1.0, v))

    parser.add_argument("--overlay-fit", type=_overlay_fit_type, default=0.75, help="Udział wysokości kadru dla panelu")
    parser.add_argument("--overlay-margin", type=int, default=0, help="Margines wokół panelu")
    parser.add_argument(
        "--overlay-mode",
        choices=["anchored", "center"],
        default="anchored",
        help="Pozycjonowanie panelu (anchored=centered to page pos, center=na środku)",
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=1.15,
        help="Mnożnik skali panelu względem lokalnej skali tła",
    )
    parser.add_argument(
        "--bg-source",
        choices=["page", "blur", "stretch", "gradient"],
        default="page",
        help="Źródło tła: page (crop strony z toningiem), blur, stretch, gradient",
    )
    parser.add_argument(
        "--bg-tone-strength",
        type=_parallax_type,
        default=0.7,
        help="Siła tonowania tła",
    )
    parser.add_argument(
        "--fg-shadow",
        type=_parallax_type,
        default=0.25,
        help="Opacity cienia pod panelem (0..1, 0 = brak cienia)",
    )
    parser.add_argument("--fg-shadow-blur", type=_clamp_nonneg_int, default=18, help="Rozmycie cienia fg")
    parser.add_argument("--fg-shadow-offset", type=_clamp_nonneg_int, default=4, help="Offset cienia fg")
    parser.add_argument(
        "--fg-shadow-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Tryb cienia foreground",
    )
    parser.add_argument("--parallax-bg", type=_parallax_type, default=0.85, help="Paralaksa tła overlay")
    parser.add_argument("--parallax-fg", type=_parallax_fg_type, default=0.0, help="Paralaksa panelu")
    parser.add_argument(
        "--gutter-thicken",
        type=_clamp_nonneg_int,
        default=2,
        help="Pogrubienie korytarzy przy eksporcie masek (px)",
    )
    parser.add_argument(
        "--min-panel-area-ratio",
        type=float,
        default=0.03,
        help="Minimalny udział panelu w stronie",
    )
    parser.add_argument(
        "--debug-overlay",
        action="store_true",
        help="Zapisz PNG z overlay dla pierwszych segmentów",
    )
    parser.add_argument("--items-from", help="Folder z maskami paneli")
    args = parser.parse_args()

    if args.mode is None:
        args.mode = "panels-overlay" if args.oneclick else "classic"

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

    if args.export_panels:
        from .panels import export_panels

        images = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        images.sort(key=lambda s: os.path.basename(s).lower())
        if not images:
            raise FileNotFoundError("Brak obrazów w folderze.")
        for i, path in enumerate(images, 1):
            out_sub = os.path.join(args.export_panels, f"page_{i:04d}")
            export_panels(
                path,
                out_sub,
                mode=args.export_mode,
                bleed=args.panel_bleed,
                tight_border=args.tight_border,
                feather=args.feather,
                gutter_thicken=args.gutter_thicken,
                min_area_ratio=args.min_panel_area_ratio,
            )
        return

    if args.oneclick:
        _run_oneclick(args, target_size)
        return

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
    elif args.mode == "panels-overlay":
        from .builder import make_panels_overlay_sequence
        from .panels import export_panels
        from .audio import extract_beats
        import tempfile

        pages_dir = args.folder
        if os.path.isdir(os.path.join(args.folder, "pages")):
            pages_dir = os.path.join(args.folder, "pages")
        page_paths = [
            os.path.join(pages_dir, f)
            for f in os.listdir(pages_dir)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        page_paths.sort(key=lambda s: os.path.basename(s).lower())
        if not page_paths:
            raise FileNotFoundError("Brak obrazów stron.")

        if args.items_from:
            panels_dir = args.items_from
        else:
            try:
                tmpd = tempfile.mkdtemp()
                for i, p in enumerate(page_paths, 1):
                    out_sub = os.path.join(tmpd, f"page_{i:04d}")
                    export_panels(
                        p,
                        out_sub,
                        mode="mask",
                        bleed=0,
                        tight_border=0,
                        feather=1,
                        gutter_thicken=args.gutter_thicken,
                        min_area_ratio=args.min_panel_area_ratio,
                    )
                panels_dir = tmpd
            except Exception as e:
                raise SystemExit(
                    "Failed to export panels to temporary directory. "
                    "Use --items-from to supply existing masks."
                ) from e

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

        clip = make_panels_overlay_sequence(
            page_paths,
            panels_dir,
            target_size=target_size,
            fps=30,
            dwell=args.dwell,
            travel=args.travel,
            travel_ease=args.travel_ease,
            overlay_fit=args.overlay_fit,
            overlay_margin=args.overlay_margin,
            overlay_mode=args.overlay_mode,
            overlay_scale=args.overlay_scale,
            bg_source=args.bg_source,
            bg_tone_strength=args.bg_tone_strength,
            parallax_bg=args.parallax_bg,
            parallax_fg=args.parallax_fg,
            fg_shadow=args.fg_shadow,
            fg_shadow_blur=args.fg_shadow_blur,
            fg_shadow_offset=args.fg_shadow_offset,
            fg_shadow_mode=args.fg_shadow_mode,
            min_panel_area_ratio=args.min_panel_area_ratio,
            gutter_thicken=args.gutter_thicken,
            debug_overlay=args.debug_overlay,
            limit_items=args.limit_items,
            trans=args.trans,
            trans_dur=args.trans_dur,
            smear_strength=args.smear_strength,
        )
        if audio_path:
            audio = _fit_audio_clip(audio_path, clip.duration, args.audio_fit)
            clip = clip.set_audio(audio)
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
    elif args.mode == "panels-items":
        from .builder import make_panels_items_sequence

        panel_paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        panel_paths.sort(key=lambda s: os.path.basename(s).lower())
        if not panel_paths:
            raise FileNotFoundError("Brak paneli w folderze.")
        clip = make_panels_items_sequence(
            panel_paths,
            target_size=target_size,
            fps=30,
            dwell=args.dwell,
            trans=args.trans,
            trans_dur=args.trans_dur,
            smear_strength=args.smear_strength,
            zoom_max=args.zoom_max,
            page_scale=args.page_scale,
            bg_mode=args.bg_mode,
            bg_parallax=args.bg_parallax,
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
