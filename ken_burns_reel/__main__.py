"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .bin_config import resolve_imagemagick, resolve_tesseract
from .builder import make_filmstrip, _export_profile, _fit_audio_clip
from .layers import shadow_cache_stats
from .ocr import verify_tesseract_available
from .validate import validate_args


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


def _legacy_out_path(output_arg: str | None, default_name: str, base_folder: str) -> str:
    if output_arg:
        if output_arg.endswith(os.sep) or os.path.isdir(output_arg):
            out = os.path.join(output_arg, default_name)
        else:
            out = output_arg
    else:
        out = os.path.join(base_folder, default_name)

    os.makedirs(os.path.dirname(out), exist_ok=True)

    if not os.path.exists(out):
        return out
    root, ext = os.path.splitext(out)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cand = f"{root}_{ts}{ext}"
    if not os.path.exists(cand):
        return cand
    i = 2
    while True:
        cand = f"{root}_{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def _resolve_out_path(args: argparse.Namespace, default_name: str, base_folder: str) -> str:
    if args.out_naming == "keep":
        return _legacy_out_path(args.output, default_name, base_folder)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = Path(args.folder).name
    mode = args.mode or "classic"
    if args.out_naming == "custom":
        prefix = args.out_prefix or "video"
        name = f"{prefix}_{ts}.mp4"
    else:
        prefix = args.out_prefix or ""
        name = f"{prefix}{slug}-{mode}_{ts}.mp4"
    out_dir = args.output if args.output else base_folder
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


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
                roughen=args.roughen,
                roughen_scale=args.roughen_scale,
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
        audio_path = args.audio
        if not audio_path and candidates:
            audio_path = candidates[0]
        if audio_path and args.align_beat:
            beat_times = extract_beats(audio_path)
        elif args.align_beat and not audio_path:
            print("⚠️ Nie znaleziono pliku audio – wideo bez wyrównania do beatów.")
            args.align_beat = False
        overlay_scale = args.overlay_scale if args.overlay_scale is not None else 1.6
        overlay_fit = args.overlay_fit if args.overlay_fit is not None else 0.7
        overlay_mode = args.overlay_mode or "anchored"

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
            overlay_fit=overlay_fit,
            overlay_mode=overlay_mode,
            overlay_scale=overlay_scale,
            bg_source="page",
            bg_blur=args.bg_blur,
            bg_tex=args.bg_tex,
            parallax_bg=args.parallax_bg,
            parallax_fg=args.parallax_fg,
            fg_shadow=args.fg_shadow,
            fg_shadow_blur=args.fg_shadow_blur,
            fg_shadow_offset=args.fg_shadow_offset,
            fg_glow=args.fg_glow,
            fg_glow_blur=args.fg_glow_blur,
            overlay_edge=args.overlay_edge,
            overlay_edge_strength=args.overlay_edge_strength,
            min_panel_area_ratio=args.min_panel_area_ratio,
            gutter_thicken=args.gutter_thicken,
            debug_overlay=args.debug_overlay,
            timing_profile=args.timing_profile,
            bpm=args.bpm,
            beats_per_panel=args.beats_per_panel,
            beats_travel=args.beats_travel,
            readability_ms=args.readability_ms,
            min_dwell=args.min_dwell,
            max_dwell=args.max_dwell,
            settle_min=args.settle_min,
            settle_max=args.settle_max,
            quantize=args.quantize,
            page_scale_overlay=args.page_scale_overlay,
            bg_vignette=args.bg_vignette,
            overlay_pop=args.overlay_pop,
            overlay_jitter=args.overlay_jitter,
            overlay_frame_px=args.overlay_frame_px,
            overlay_frame_color=args.overlay_frame_color,
            bg_offset=args.bg_offset,
            fg_offset=args.fg_offset,
            seed=args.seed,
            bubble_lift=getattr(args, "bubble_lift", False),
            detect_bubbles=args.detect_bubbles == "on",
            bubble_min_area=args.bubble_min_area,
            bubble_roundness_min=args.bubble_roundness_min,
            bubble_feather_px=getattr(args, "bubble_feather_px", 2),
            bg_drift_zoom=getattr(args, "bg_drift_zoom", 0.0),
            bg_drift_speed=getattr(args, "bg_drift_speed", 0.0),
            fg_drift_zoom=getattr(args, "fg_drift_zoom", 0.0),
            fg_drift_speed=getattr(args, "fg_drift_speed", 0.0),
            travel_path=args.travel_path,
            deep_bottom_glow=args.deep_bottom_glow,
            look=args.look,
            limit_items=args.limit_items,
            trans="smear",
            trans_dur=0.30,
            smear_strength=1.1,
        )

        if audio_path:
            audio = _fit_audio_clip(audio_path, clip.duration, args.audio_fit, gain_db=args.audio_gain)
            clip = clip.set_audio(audio)

        out_path = _resolve_out_path(args, "final_video.mp4", args.folder)
        prof = _export_profile(args.profile, args.codec, target_size)
        if prof.get("resize"):
            clip = clip.resize(newsize=prof["resize"])
        if args.profile == "perf":
            hits, misses = shadow_cache_stats()
            total = hits + misses
            rate = hits / total if total else 0.0
            logging.info(
                "shadow_cache hit-rate: %.2f%% (%d/%d)", rate * 100, hits, total
            )
            clip_count = sum(1 for _ in clip.iter_clips())
            logging.info("VideoClip count: %d", clip_count)
        clip.write_videofile(
            out_path,
            fps=prof["fps"],
            codec=prof["codec"],
            audio_codec=prof["audio_codec"],
            audio_bitrate=prof["audio_bitrate"],
            ffmpeg_params=prof["ffmpeg_params"],
            preset=prof["preset"],
        )

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Ken Burns style video")
    parser.add_argument("folder", help="Input folder with images and audio")
    parser.add_argument("--preset", action="append", default=[], help="Path to YAML preset overriding defaults")
    parser.add_argument("--tesseract", help="Path to Tesseract binary")
    parser.add_argument("--magick", help="Path to ImageMagick binary")
    parser.add_argument(
        "--output",
        help="Path to MP4 file or output directory. If existing, a timestamp/counter is appended.",
    )
    parser.add_argument(
        "--out-naming",
        choices=["auto", "keep", "custom"],
        default="auto",
        help="Naming policy for output file",
    )
    parser.add_argument("--out-prefix", default="", help="Prefix for auto/custom naming")
    parser.add_argument("--export-panels", help="Export detected panels to folder")
    parser.add_argument("--oneclick", action="store_true", help="Tryb one-click: auto video from pages and audio")
    parser.add_argument("--validate", action="store_true", help="Validate arguments and exit")
    parser.add_argument("--deterministic", action="store_true", help="Force deterministic build")
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
        "--roughen",
        type=float,
        default=0.15,
        help="Nieregularność krawędzi maski (0..1)",
    )
    parser.add_argument(
        "--roughen-scale",
        type=int,
        default=24,
        help="Skala szumu dla roughen",
    )
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
    parser.add_argument(
        "--transition-duration",
        dest="trans_dur",
        type=float,
        default=0.3,
        help="Długość przejścia/crossfadu między panelami (s)",
    )
    parser.add_argument(
        "--trans-dur",
        dest="trans_dur",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--xfade",
        dest="trans_dur",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
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
    parser.add_argument("--audio", help="Ścieżka do pliku audio")
    parser.add_argument(
        "--audio-gain",
        type=float,
        default=0.0,
        help="Wzmocnienie ścieżki audio (dB)",
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
    parser.add_argument("--bg-blur", type=float, default=8.0, help="Rozmycie tła")
    parser.add_argument(
        "--bg-tex",
        choices=["vignette", "gradient", "none"],
        default="vignette",
        help="Tekstura tła",
    )
    parser.add_argument(
        "--overlay-edge",
        choices=["feather", "torn"],
        default="feather",
        help="Typ krawędzi panelu overlay",
    )
    parser.add_argument(
        "--overlay-edge-strength",
        type=float,
        default=0.6,
        help="Siła efektu krawędzi overlay",
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
        default=None,
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
        "--smear-strength",
        type=float,
        default=1.0,
        help="Siła smuga dla przejścia smear",
    )
    parser.add_argument(
        "--profile",
        choices=["preview", "social", "quality", "perf"],
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

    parser.add_argument("--overlay-fit", type=_overlay_fit_type, default=None, help="Udział wysokości kadru dla panelu")
    parser.add_argument("--overlay-margin", type=int, default=None, help="Margines wokół panelu")
    parser.add_argument(
        "--overlay-mode",
        choices=["anchored", "center"],
        default="anchored",
        help="Pozycjonowanie panelu (anchored=centered to page pos, center=na środku)",
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=None,
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
        default=None,
        help="Opacity cienia pod panelem (0..1, 0 = brak cienia)",
    )
    parser.add_argument(
        "--fg-shadow-blur",
        type=_clamp_nonneg_int,
        default=None,
        help="Rozmycie cienia fg",
    )
    parser.add_argument(
        "--fg-shadow-offset",
        type=_clamp_nonneg_int,
        default=None,
        help="Offset cienia fg",
    )
    parser.add_argument(
        "--fg-glow",
        type=_parallax_type,
        default=None,
        help="Siła poświaty panelu",
    )
    parser.add_argument(
        "--fg-glow-blur",
        type=_clamp_nonneg_int,
        default=None,
        help="Rozmycie poświaty",
    )
    parser.add_argument(
        "--fg-shadow-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Tryb cienia foreground",
    )
    parser.add_argument("--parallax-bg", type=_parallax_type, default=None, help="Paralaksa tła overlay")
    parser.add_argument("--parallax-fg", type=_parallax_fg_type, default=None, help="Paralaksa panelu")
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
    parser.add_argument("--timing-profile", choices=["human", "music", "free"], default=None, help="Profil czasu trwania segmentów")
    parser.add_argument("--bpm", type=int, help="Ustaw tempo utworu (beats per minute)")
    parser.add_argument("--beats-per-panel", type=float, default=2.0, help="Ile beatów na panel")
    parser.add_argument("--beats-travel", type=float, default=0.5, help="Ile beatów przejazdu")
    parser.add_argument(
        "--readability-ms",
        type=int,
        default=1400,
        help="Minimalna ekspozycja panelu (ms)",
    )
    parser.add_argument("--min-dwell", type=float, default=1.0, help="Minimalny czas zatrzymania (s)")
    parser.add_argument("--max-dwell", type=float, default=1.8, help="Maksymalny czas zatrzymania (s)")
    parser.add_argument("--settle-min", type=float, default=0.12, help="Minimalny czas settle (s)")
    parser.add_argument("--settle-max", type=float, default=0.22, help="Maksymalny czas settle (s)")
    parser.add_argument("--quantize", choices=["off", "1/8", "1/4"], default="off", help="Przyciągaj starty do siatki nut")
    parser.add_argument("--overlay-pop", "--pop-scale", dest="overlay_pop", type=float, default=1.0, help="Początkowa skala overlay dla efektu pop-in")
    parser.add_argument("--overlay-jitter", "--jitter", dest="overlay_jitter", type=float, default=0.0, help="Subtelny mikro-ruch overlay (px)")
    parser.add_argument("--overlay-frame-px", type=_clamp_nonneg_int, default=0, help="Grubość ramki overlay (px)")
    parser.add_argument("--overlay-frame-color", default="#000000", help="Kolor ramki overlay w formacie #RRGGBB")
    parser.add_argument("--detect-bubbles", choices=["on", "off"], default="off")
    parser.add_argument("--bubble-mode", choices=["mask", "rect"], default="mask")
    parser.add_argument("--bubble-min-area", type=int, default=200)
    parser.add_argument("--bubble-roundness-min", type=float, default=0.3)
    parser.add_argument("--bubble-contrast-min", type=float, default=0.0)
    parser.add_argument("--bubble-export", choices=["none", "masks", "keyframes"], default="none")
    parser.add_argument("--bg-offset", type=float, default=0.0, help="Opóźnienie ruchu tła (s)")
    parser.add_argument("--fg-offset", type=float, default=0.0, help="Opóźnienie ruchu panelu (s)")
    parser.add_argument("--seed", type=int, default=0, help="Seed deterministycznego driftu")
    parser.add_argument("--travel-path", choices=["linear", "arc"], default="linear", help="Tor przejazdu kamery")
    parser.add_argument("--deep-bottom-glow", type=float, default=0.0, help="Poświata od dołu (0..1)")
    parser.add_argument("--page-scale-overlay", type=_page_scale_type, default=1.0, help="Skala strony przy overlay")
    parser.add_argument("--bg-vignette", type=_parallax_type, default=0.15, help="Siła winiety tła")
    parser.add_argument("--look", choices=["none", "witcher1"], default="none", help="Preset koloru tła")
    parser.add_argument("--items-from", help="Folder z maskami paneli")

    prelim, _ = parser.parse_known_args(argv)
    preset_paths = prelim.preset
    style_presets = [p for p in preset_paths if "styles" in p]
    motion_presets = [p for p in preset_paths if "motion" in p or "motions" in p]
    other_presets = [p for p in preset_paths if p not in style_presets + motion_presets]
    for path in style_presets + motion_presets + other_presets:
        with open(path, "r", encoding="utf8") as fh:
            data = yaml.safe_load(fh) or {}
        parser.set_defaults(**data)

    args = parser.parse_args(argv)
    if argv and "--xfade" in argv:
        print("⚠️ --xfade is deprecated, use --transition-duration", file=sys.stderr)

    if isinstance(getattr(args, "bubbles", None), dict):
        bconf = args.bubbles
        if "lift" in bconf:
            args.bubble_lift = bool(bconf.get("lift"))
        if "feather_px" in bconf:
            args.bubble_feather_px = int(bconf.get("feather_px"))
    if args.mode is None:
        args.mode = "panels-overlay" if args.oneclick else "classic"

    if args.timing_profile is None:
        args.timing_profile = (
            "human" if (args.oneclick or args.mode == "panels-overlay") else "free"
        )

    if args.bg_parallax is None and getattr(args, "parallax_bg", None) is not None:
        args.bg_parallax = args.parallax_bg
    if args.bg_parallax is None:
        args.bg_parallax = 0.05 if args.mode == "panels-overlay" else 0.85
    if args.parallax_bg is None:
        args.parallax_bg = args.bg_parallax
    if args.parallax_fg is None:
        args.parallax_fg = 0.0
    if args.overlay_fit is None:
        args.overlay_fit = 0.72 if args.mode == "panels-overlay" else 0.75
    if args.overlay_margin is None:
        args.overlay_margin = 24 if args.mode == "panels-overlay" else 0
    if args.fg_shadow is None:
        args.fg_shadow = 0.22 if args.mode == "panels-overlay" else 0.25
    if args.fg_shadow_blur is None:
        args.fg_shadow_blur = 14 if args.mode == "panels-overlay" else 18
    if args.fg_shadow_offset is None:
        args.fg_shadow_offset = 4
    if args.fg_glow is None:
        args.fg_glow = 0.10 if args.mode == "panels-overlay" else 0.0
    if args.fg_glow_blur is None:
        args.fg_glow_blur = 24
    # łagodniejsze powiększenie paneli
    if args.mode == "panels-overlay" and (
        args.overlay_scale is None or args.overlay_scale == 1.6
    ):
        args.overlay_scale = 1.45

    if args.preview:
        args.profile = "preview"

    if args.travel_ease == "ease":
        args.travel_ease = "inout"
    args.readability_ms = max(args.readability_ms, 1400)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logging.info("deterministic build seed=%s", args.seed)
    if args.validate:
        errs = validate_args(args)
        if errs:
            for e in errs:
                print(f"validation error: {e}", file=sys.stderr)
            raise SystemExit(1)
        return
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
                roughen=args.roughen,
                roughen_scale=args.roughen_scale,
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
        audio_path = args.audio
        if not audio_path and audios:
            audio_path = os.path.join(args.folder, audios[0])
        if audio_path and args.align_beat:
            beat_times = extract_beats(audio_path)

        clip = make_panels_cam_sequence(
            images,
            target_size=target_size,
            fps=30,
            dwell=args.dwell,
            travel=args.travel,
            trans_dur=args.trans_dur,
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
        out_path = _resolve_out_path(args, "final_video.mp4", args.folder)
        prof = _export_profile(args.profile, args.codec, target_size)
        if prof.get("resize"):
            clip = clip.resize(newsize=prof["resize"])
        if args.profile == "perf":
            hits, misses = shadow_cache_stats()
            total = hits + misses
            rate = hits / total if total else 0.0
            logging.info(
                "shadow_cache hit-rate: %.2f%% (%d/%d)", rate * 100, hits, total
            )
            clip_count = sum(1 for _ in clip.iter_clips())
            logging.info("VideoClip count: %d", clip_count)
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
                        roughen=args.roughen,
                        roughen_scale=args.roughen_scale,
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
            align_beat=args.align_beat,
            beat_times=beat_times,
            overlay_fit=args.overlay_fit,
            overlay_margin=args.overlay_margin,
            overlay_mode=args.overlay_mode,
            overlay_scale=args.overlay_scale,
            bg_source=args.bg_source,
            bg_blur=args.bg_blur,
            bg_tex=args.bg_tex,
            bg_tone_strength=args.bg_tone_strength,
            parallax_bg=args.parallax_bg,
            parallax_fg=args.parallax_fg,
            fg_shadow=args.fg_shadow,
            fg_shadow_blur=args.fg_shadow_blur,
            fg_shadow_offset=args.fg_shadow_offset,
            fg_shadow_mode=args.fg_shadow_mode,
            fg_glow=args.fg_glow,
            fg_glow_blur=args.fg_glow_blur,
            overlay_edge=args.overlay_edge,
            overlay_edge_strength=args.overlay_edge_strength,
            min_panel_area_ratio=args.min_panel_area_ratio,
            gutter_thicken=args.gutter_thicken,
            debug_overlay=args.debug_overlay,
            timing_profile=args.timing_profile,
            bpm=args.bpm,
            beats_per_panel=args.beats_per_panel,
            beats_travel=args.beats_travel,
            readability_ms=args.readability_ms,
            min_dwell=args.min_dwell,
            max_dwell=args.max_dwell,
            settle_min=args.settle_min,
            settle_max=args.settle_max,
            quantize=args.quantize,
            page_scale_overlay=args.page_scale_overlay,
            bg_vignette=args.bg_vignette,
            overlay_pop=args.overlay_pop,
            overlay_jitter=args.overlay_jitter,
            overlay_frame_px=args.overlay_frame_px,
            overlay_frame_color=args.overlay_frame_color,
            bg_offset=args.bg_offset,
            fg_offset=args.fg_offset,
            seed=args.seed,
            bubble_lift=getattr(args, "bubble_lift", False),
            detect_bubbles=args.detect_bubbles == "on",
            bubble_min_area=args.bubble_min_area,
            bubble_roundness_min=args.bubble_roundness_min,
            bubble_feather_px=getattr(args, "bubble_feather_px", 2),
            bg_drift_zoom=getattr(args, "bg_drift_zoom", 0.0),
            bg_drift_speed=getattr(args, "bg_drift_speed", 0.0),
            fg_drift_zoom=getattr(args, "fg_drift_zoom", 0.0),
            fg_drift_speed=getattr(args, "fg_drift_speed", 0.0),
            travel_path=args.travel_path,
            deep_bottom_glow=args.deep_bottom_glow,
            look=args.look,
            limit_items=args.limit_items,
            trans=args.trans,
            trans_dur=args.trans_dur,
            smear_strength=args.smear_strength,
        )
        if audio_path:
            audio = _fit_audio_clip(audio_path, clip.duration, args.audio_fit, gain_db=args.audio_gain)
            clip = clip.set_audio(audio)
        out_path = _resolve_out_path(args, "final_video.mp4", args.folder)
        prof = _export_profile(args.profile, args.codec, target_size)
        if prof.get("resize"):
            clip = clip.resize(newsize=prof["resize"])
        if args.profile == "perf":
            hits, misses = shadow_cache_stats()
            total = hits + misses
            rate = hits / total if total else 0.0
            logging.info(
                "shadow_cache hit-rate: %.2f%% (%d/%d)", rate * 100, hits, total
            )
            clip_count = sum(1 for _ in clip.iter_clips())
            logging.info("VideoClip count: %d", clip_count)
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
        out_path = _resolve_out_path(args, "final_video.mp4", args.folder)
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
        audio_exts = {".mp3", ".wav", ".m4a"}
        has_audio = any(
            os.path.splitext(f)[1].lower() in audio_exts for f in os.listdir(args.folder)
        )
        audio_fit = args.audio_fit
        if not has_audio and not args.audio:
            audio_fit = "silence"
        src = make_filmstrip(
            args.folder,
            audio_fit=audio_fit,
            profile=args.profile,
            codec=args.codec,
            target_size=target_size,
        )
        if os.path.exists(src):
            dst = _resolve_out_path(args, "final_video.mp4", args.folder)
            if os.path.abspath(src) != os.path.abspath(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.replace(src, dst)


if __name__ == "__main__":
    main()
