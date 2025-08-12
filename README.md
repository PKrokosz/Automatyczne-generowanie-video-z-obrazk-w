# Ken Burns Reel

Ken Burns Reel is a modular video assembly engine for turning static pages or panels into dynamic videos. It supports camera motion, panel detection, overlays, and style presets, enabling automated reels with the Ken Burns effect, parallax backgrounds, and configurable transitions.

## Features
- Ken Burns zoom and pan across comic panels.
- Panel detection with mask export.
- Multiple composition modes: classic filmstrip, panel sequences, item overlays.
- Style and motion presets via YAML.
- Arc-path camera travel and subtle background/foreground drift.
- Caption rendering with Tesseract and ImageMagick.
- Audio alignment and beat extraction.
- Transition effects (smear, slide, whip-pan, overlay lift).

## Installation
### Requirements
- Python 3.10–3.12
- FFmpeg (MoviePy backend)
- [ImageMagick](https://imagemagick.org) (`magick` CLI)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`tesseract` CLI)

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev] || pip install -r requirements.txt -r requirements-dev.txt
```
Ensure `magick` and `tesseract` binaries are on `PATH` or configured via environment variables.

### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .[dev]  # or use requirements files
```
Install ImageMagick and Tesseract and add their directories to `PATH`. Use the unified `magick` wrapper and verify `tesseract.exe` is reachable.

## Quick Start
Render a filmstrip of panels with a style preset:

**Bash**
```bash
python -m ken_burns_reel input_folder --preset styles/float_black_v1.yaml \
  --mode panels-overlay --transition-duration 0.3
```

**PowerShell**
```powershell
python -m ken_burns_reel input_folder `
  --preset styles/float_black_v1.yaml `
  --mode panels-overlay `
  --transition-duration 0.3
```

**CMD**
```cmd
python -m ken_burns_reel input_folder ^
  --preset styles\float_black_v1.yaml ^
  --mode panels-overlay ^
  --transition-duration 0.3
```

## Pipeline
1. **Import** – images and optional audio are loaded.
2. **Panel Detection** – `panels.py` locates panels and exports masks if requested.
3. **Motion & Composition** – `builder.py` assembles sequences, applying camera motion (`motion.py`), transitions (`transitions.py`), layers (`layers.py`), and optional captions (`captions.py`).
4. **Export** – `moviepy` renders the final video according to profile and codec settings.

## Modes and Presets
`--mode` selects the assembly strategy:
- `classic` – legacy filmstrip exporter.
- `panels` – camera moves across detected panels.
- `panels-items` – montage from individual panel images.
- `panels-overlay` – page as background, panel as foreground overlay.

`--preset` loads YAML files overriding defaults. Style presets (e.g., `styles/float_black_v1.yaml`) and motion presets may be combined; later presets override earlier ones.

## Full CLI Flags
Below is the complete list of command-line options for `python -m ken_burns_reel`:

| Flags | Dest | Default | Help |
|------|------|---------|------|
| `folder` | `` | `None` | Input folder with images and audio |
| `--preset` | `` | `[]` | Path to YAML preset overriding defaults |
| `--tesseract` | `` | `None` | Path to Tesseract binary |
| `--magick` | `` | `None` | Path to ImageMagick binary |
| `--output` | `` | `None` | Path to MP4 file or output directory. If existing, a timestamp/counter is appended. |
| `--out-naming` | `` | `auto` | Naming policy for output file |
| `--out-prefix` | `` | `` | Prefix for auto/custom naming |
| `--export-panels` | `` | `None` | Export detected panels to folder |
| `--oneclick` | `` | `None` | Tryb one-click: auto video from pages and audio |
| `--validate` | `` | `None` | Validate arguments and exit |
| `--deterministic` | `` | `None` | Force deterministic build |
| `--export-mode` | `` | `rect` | Panel export mode |
| `--mode` | `` | `None` | classic: dotychczasowy montaż; panels: ruch kamery po panelach komiksu; panels-items: montaż z pojedynczych paneli; panels-overlay: tło strona, foreground panel |
| `--limit-items` | `` | `999` | Limit liczby paneli w overlay |
| `--tight-border` | `` | `1` | Erozja konturu w eksporcie mask (px) |
| `--feather` | `` | `1` | Feather alpha w eksporcie mask (px) |
| `--roughen` | `` | `0.15` | Nieregularność krawędzi maski (0..1) |
| `--roughen-scale` | `` | `24` | Skala szumu dla roughen |
| `--enhance` | `` | `comic` | Tryb poprawy paneli |
| `--enhance-strength` | `` | `1.0` | Siła enhance |
| `--shadow` | `` | `0.2` | Opacity cienia pod panelem |
| `--shadow-blur` | `` | `12` | Rozmycie cienia (px) |
| `--shadow-offset` | `` | `3` | Offset cienia (px) |
| `--dwell` | `` | `1.0` | Czas zatrzymania na panelu (s) |
| `--travel` | `` | `0.6` | Czas przejazdu między panelami (s) |
| `--transition-duration` | `trans_dur` | `0.3` | Długość przejścia/crossfadu między panelami (s) |
| `--trans-dur` | `trans_dur` | `None` |  |
| `--xfade` | `trans_dur` | `None` |  |
| `--settle` | `` | `0.14` | Długość micro-holdu (s) |
| `--travel-ease, --easing` | `travel_ease` | `inout` | Profil jazdy kamery |
| `--dwell-scale` | `` | `1.0` | Globalne skalowanie czasu zatrzymania po zważeniu |
| `--align-beat` | `` | `None` | Wyrównaj start stron do najbliższego beatu |
| `--debug-panels` | `` | `None` | Tryb debug – zapisuje plik panels_debug.jpg z wykrytymi ramkami i kończy działanie. |
| `--audio-fit` | `` | `trim` | Jak dopasować audio do długości wideo |
| `--audio` | `` | `None` | Ścieżka do pliku audio |
| `--audio-gain` | `` | `0.0` | Wzmocnienie ścieżki audio (dB) |
| `--dwell-mode` | `` | `first` | Na ilu panelach zatrzymywać się w pełni |
| `--bg-mode` | `` | `blur` | Underlay pod stroną |
| `--bg-blur` | `` | `8.0` | Rozmycie tła |
| `--bg-tex` | `` | `vignette` | Tekstura tła |
| `--overlay-edge` | `` | `feather` | Typ krawędzi panelu overlay |
| `--overlay-edge-strength` | `` | `0.6` | Siła efektu krawędzi overlay |
| `--page-scale` | `` | `0.92` | Skala foreground (mniejsza niż 1.0 = widać tło) |
| `--bg-parallax` | `` | `None` | Siła paralaksy tła podczas travelu |
| `--panel-bleed` | `` | `24` | Margines przy kadrowaniu panelu (px) |
| `--zoom-max` | `` | `1.06` | Maksymalne dodatkowe przybliżenie dla małego tekstu |
| `--trans` | `` | `smear` | Przejście między panelami w trybie panels-items |
| `--smear-strength` | `` | `1.0` | Siła smuga dla przejścia smear |
| `--profile` | `` | `social` | Preset eksportu |
| `--preview` | `` | `None` | Skrót dla --profile preview |
| `--codec` | `` | `h264` | Kodek wideo |
| `--size` | `` | `None` | Docelowy rozmiar WxH |
| `--aspect` | `` | `None` | Proporcje (z --height) |
| `--height` | `` | `None` | Wysokość dla --aspect |
| `--overlay-fit` | `` | `None` | Udział wysokości kadru dla panelu |
| `--overlay-margin` | `` | `None` | Margines wokół panelu |
| `--overlay-mode` | `` | `anchored` | Pozycjonowanie panelu (anchored=centered to page pos, center=na środku) |
| `--overlay-scale` | `` | `None` | Mnożnik skali panelu względem lokalnej skali tła |
| `--bg-source` | `` | `page` | Źródło tła: page (crop strony z toningiem), blur, stretch, gradient |
| `--bg-tone-strength` | `` | `0.7` | Siła tonowania tła |
| `--fg-shadow` | `` | `None` | Opacity cienia pod panelem (0..1, 0 = brak cienia) |
| `--fg-shadow-blur` | `` | `None` | Rozmycie cienia fg |
| `--fg-shadow-offset` | `` | `None` | Offset cienia fg |
| `--fg-glow` | `` | `None` | Siła poświaty panelu |
| `--fg-glow-blur` | `` | `None` | Rozmycie poświaty |
| `--fg-shadow-mode` | `` | `soft` | Tryb cienia foreground |
| `--parallax-bg` | `` | `None` | Paralaksa tła overlay |
| `--parallax-fg` | `` | `None` | Paralaksa panelu |
| `--gutter-thicken` | `` | `2` | Pogrubienie korytarzy przy eksporcie masek (px) |
| `--min-panel-area-ratio` | `` | `0.03` | Minimalny udział panelu w stronie |
| `--debug-overlay` | `` | `None` | Zapisz PNG z overlay dla pierwszych segmentów |
| `--timing-profile` | `` | `None` | Profil czasu trwania segmentów |
| `--bpm` | `` | `None` | Ustaw tempo utworu (beats per minute) |
| `--beats-per-panel` | `` | `2.0` | Ile beatów na panel |
| `--beats-travel` | `` | `0.5` | Ile beatów przejazdu |
| `--readability-ms` | `` | `1400` | Minimalna ekspozycja panelu (ms) |
| `--min-dwell` | `` | `1.0` | Minimalny czas zatrzymania (s) |
| `--max-dwell` | `` | `1.8` | Maksymalny czas zatrzymania (s) |
| `--settle-min` | `` | `0.12` | Minimalny czas settle (s) |
| `--settle-max` | `` | `0.22` | Maksymalny czas settle (s) |
| `--quantize` | `` | `off` | Przyciągaj starty do siatki nut |
| `--overlay-pop, --pop-scale` | `overlay_pop` | `1.0` | Początkowa skala overlay dla efektu pop-in |
| `--overlay-jitter, --jitter` | `overlay_jitter` | `0.0` | Subtelny mikro-ruch overlay (px) |
| `--overlay-frame-px` | `` | `0` | Grubość ramki overlay (px) |
| `--overlay-frame-color` | `` | `#000000` | Kolor ramki overlay w formacie #RRGGBB |
| `--bg-offset` | `` | `0.0` | Opóźnienie ruchu tła (s) |
| `--fg-offset` | `` | `0.0` | Opóźnienie ruchu panelu (s) |
| `--seed` | `` | `0` | Seed deterministycznego driftu |
| `--travel-path` | `` | `linear` | Tor przejazdu kamery |
| `--deep-bottom-glow` | `` | `0.0` | Poświata od dołu (0..1) |
| `--page-scale-overlay` | `` | `1.0` | Skala strony przy overlay |
| `--bg-vignette` | `` | `0.15` | Siła winiety tła |
| `--look` | `` | `none` | Preset koloru tła |
| `--items-from` | `` | `None` | Folder z maskami paneli |

## Advanced Options and Validation
- `--validate` parses arguments and exits after reporting validation errors.
- `--deterministic` seeds Python and NumPy RNGs (use with `--seed`) for reproducible motion.
- Caching: `layers.page_shadow` caches generated shadows; cache stats are logged in `--profile perf`.
- Tests use golden images and JSON logs; the `--look` option in `tests/test_overlay_lift.py` writes reference frames for docs/LOOK.md.

## Tests and CI
Run the full test suite:
```bash
pytest -q
```
Some tests depend on ImageMagick or Tesseract; they are skipped when binaries are unresolved. Golden files reside under `tests/` and are compared during CI.

## Module Overview
The package is organised as follows:

### __init__.py
- Functions: make_filmstrip

### __main__.py
- Functions: _page_scale_type, _parallax_type, _parallax_fg_type, _nonneg_int, _clamp_nonneg_int, _zoom_max_type, _legacy_out_path, _resolve_out_path, _run_oneclick, parse_args, main

### audio.py
- Functions: extract_beats

### bin_config.py
- Functions: _validate_binary, resolve_imagemagick, resolve_tesseract

### builder.py
- Functions: _fit_audio_clip, _fit_window_to_box, _interp, ease_in_out, ease_in, ease_out, _apply_witcher_look, _get_ease_fn, _with_duration, apply_clahe_rgb, enhance_panel, _paste_rgba_clipped, _hex_to_rgb, _zoom_image_center, _darken_region_with_alpha_clipped, _add_rgb_clipped, _attach_mask, _make_underlay, make_panels_cam_clip, make_panels_cam_sequence, make_panels_items_sequence, compute_segment_timing, make_panels_overlay_sequence, _export_profile, ken_burns_scroll, make_filmstrip

### captions.py
- Functions: sanitize_caption, is_caption_meaningful, render_caption_clip, overlay_caption

### cli.py
- Functions: build_parser, validate_args, main

### color.py
- Functions: srgb_to_linear16, linear16_to_srgb

### config.py

### focus.py
- Functions: detect_focus_point

### layers.py
- Functions: _premultiply, _paste_rgba, page_shadow, shadow_cache_stats

### motion.py
- Classes: DriftParams
- Functions: arc_path, subtle_drift, apply_transform

### ocr.py
- Functions: extract_caption, verify_tesseract_available, page_ocr_data, text_boxes_stats

### panels.py
- Functions: _build_panels_mask, alpha_bbox, fill_holes, roughen_alpha, detect_panels, _suppress_nested, order_panels_lr_tb, export_panels, debug_detect_panels

### transitions.py
- Functions: ease_in_out, ease_in, ease_out, _get_ease_fn, slide_transition, smear_transition, whip_pan_transition, fg_fade, smear_bg_crossfade_fg, overlay_lift

### utils.py
- Functions: smart_crop, gaussian_blur, _set_fps

## Troubleshooting
- **Tesseract not found**: set `TESSERACT_BINARY` or use `--tesseract /path/to/tesseract`.
- **ImageMagick errors**: ensure `magick` exists and security policies permit `label:` usage.
- **Missing libGL**: install system package `libgl1` for OpenCV headless environments.
- **Caption rendering skipped**: if ImageMagick is unavailable, captions are ignored with a warning.

## References
- Style presets: `styles/`
- Example CLI invocations: `docs/cli_examples.md`
- Visualization helpers and golden frame generation: `docs/LOOK.md`
