# CLI Reference
| Flag | Type | Default | Help |
|------|------|---------|------|
| folder | str | None | Input folder with images and audio |
| --preset | str | [] | Path to YAML preset overriding defaults |
| --tesseract | str | None | Path to Tesseract binary |
| --magick | str | None | Path to ImageMagick binary |
| --output | str | None | Path to MP4 file or output directory. If existing, a timestamp/counter is appended. |
| --out-naming | str | auto | Naming policy for output file |
| --out-prefix | str |  | Prefix for auto/custom naming |
| --export-panels | str | None | Export detected panels to folder |
| --oneclick | str | None | Tryb one-click: auto video from pages and audio |
| --validate | str | None | Validate arguments and exit |
| --deterministic | str | None | Force deterministic build |
| --export-mode | str | rect | Panel export mode |
| --mode | str | None | classic: dotychczasowy montaż; panels: ruch kamery po panelach komiksu; panels-items: montaż z pojedynczych paneli; panels-overlay: tło strona, foreground panel |
| --limit-items | int | 999 | Limit liczby paneli w overlay |
| --tight-border | int | 1 | Erozja konturu w eksporcie mask (px) |
| --feather | int | 1 | Feather alpha w eksporcie mask (px) |
| --roughen | float | 0.15 | Nieregularność krawędzi maski (0..1) |
| --roughen-scale | int | 24 | Skala szumu dla roughen |
| --enhance | str | comic | Tryb poprawy paneli |
| --enhance-strength | float | 1.0 | Siła enhance |
| --shadow | _parallax_type | 0.2 | Opacity cienia pod panelem |
| --shadow-blur | _clamp_nonneg_int | 12 | Rozmycie cienia (px) |
| --shadow-offset | _clamp_nonneg_int | 3 | Offset cienia (px) |
| --dwell | float | 1.0 | Czas zatrzymania na panelu (s) |
| --travel | float | 0.6 | Czas przejazdu między panelami (s) |
| --transition-duration | float | 0.3 | Długość przejścia/crossfadu między panelami (s) |
| --trans-dur | float | argparse.SUPPRESS |  |
| --xfade | float | argparse.SUPPRESS |  |
| --settle | float | 0.14 | Długość micro-holdu (s) |
| --travel-ease | str | inout | Profil jazdy kamery |
| --dwell-scale | float | 1.0 | Globalne skalowanie czasu zatrzymania po zważeniu |
| --align-beat | str | None | Wyrównaj start stron do najbliższego beatu |
| --debug-panels | str | None | Tryb debug – zapisuje plik panels_debug.jpg z wykrytymi ramkami i kończy działanie. |
| --audio-fit | str | trim | Jak dopasować audio do długości wideo |
| --audio | str | None | Ścieżka do pliku audio |
| --audio-gain | float | 0.0 | Wzmocnienie ścieżki audio (dB) |
| --dwell-mode | str | first | Na ilu panelach zatrzymywać się w pełni |
| --bg-mode | str | blur | Underlay pod stroną |
| --bg-blur | float | 8.0 | Rozmycie tła |
| --bg-tex | str | vignette | Tekstura tła |
| --overlay-edge | str | feather | Typ krawędzi panelu overlay |
| --overlay-edge-strength | float | 0.6 | Siła efektu krawędzi overlay |
| --page-scale | _page_scale_type | 0.92 | Skala foreground (mniejsza niż 1.0 = widać tło) |
| --bg-parallax | _parallax_type | None | Siła paralaksy tła podczas travelu |
| --panel-bleed | _nonneg_int | 24 | Margines przy kadrowaniu panelu (px) |
| --zoom-max | _zoom_max_type | 1.06 | Maksymalne dodatkowe przybliżenie dla małego tekstu |
| --trans | str | smear | Przejście między panelami w trybie panels-items |
| --smear-strength | float | 1.0 | Siła smuga dla przejścia smear |
| --profile | str | social | Preset eksportu |
| --preview | str | None | Skrót dla --profile preview |
| --codec | str | h264 | Kodek wideo |
| --size | str | None | Docelowy rozmiar WxH |
| --aspect | str | choices=['9:16', '16:9', '1:1'] | Proporcje (z --height) |
| --height | int | None | Wysokość dla --aspect |
| --overlay-fit | _overlay_fit_type | None | Udział wysokości kadru dla panelu |
| --overlay-margin | int | None | Margines wokół panelu |
| --overlay-mode | str | anchored | Pozycjonowanie panelu (anchored=centered to page pos, center=na środku) |
| --overlay-scale | float | None | Mnożnik skali panelu względem lokalnej skali tła |
| --bg-source | str | page | Źródło tła: page (crop strony z toningiem), blur, stretch, gradient |
| --bg-tone-strength | _parallax_type | 0.7 | Siła tonowania tła |
| --fg-shadow | _parallax_type | None | Opacity cienia pod panelem (0..1, 0 = brak cienia) |
| --fg-shadow-blur | _clamp_nonneg_int | None | Rozmycie cienia fg |
| --fg-shadow-offset | _clamp_nonneg_int | None | Offset cienia fg |
| --fg-glow | _parallax_type | None | Siła poświaty panelu |
| --fg-glow-blur | _clamp_nonneg_int | None | Rozmycie poświaty |
| --fg-shadow-mode | str | soft | Tryb cienia foreground |
| --parallax-bg | _parallax_type | None | Paralaksa tła overlay |
| --parallax-fg | _parallax_fg_type | None | Paralaksa panelu |
| --gutter-thicken | _clamp_nonneg_int | 2 | Pogrubienie korytarzy przy eksporcie masek (px) |
| --min-panel-area-ratio | float | 0.03 | Minimalny udział panelu w stronie |
| --debug-overlay | str | None | Zapisz PNG z overlay dla pierwszych segmentów |
| --timing-profile | str | None | Profil czasu trwania segmentów |
| --bpm | int | None | Ustaw tempo utworu (beats per minute) |
| --beats-per-panel | float | 2.0 | Ile beatów na panel |
| --beats-travel | float | 0.5 | Ile beatów przejazdu |
| --readability-ms | int | 1400 | Minimalna ekspozycja panelu (ms, min 1400) |
| --min-dwell | float | 1.0 | Minimalny czas zatrzymania (s) |
| --max-dwell | float | 1.8 | Maksymalny czas zatrzymania (s) |
| --settle-min | float | 0.12 | Minimalny czas settle (s) |
| --settle-max | float | 0.22 | Maksymalny czas settle (s) |
| --quantize | str | off | Przyciągaj starty do siatki nut |
| --overlay-pop | float | 1.0 | Początkowa skala overlay dla efektu pop-in |
| --overlay-jitter | float | 0.0 | Subtelny mikro-ruch overlay (px) |
| --overlay-frame-px | _clamp_nonneg_int | 0 | Grubość ramki overlay (px) |
| --overlay-frame-color | str | #000000 | Kolor ramki overlay w formacie #RRGGBB |
| --bg-offset | float | 0.0 | Opóźnienie ruchu tła (s) |
| --fg-offset | float | 0.0 | Opóźnienie ruchu panelu (s) |
| --seed | int | 0 | Seed deterministycznego driftu |
| --travel-path | str | linear | Tor przejazdu kamery |
| --deep-bottom-glow | float | 0.0 | Poświata od dołu (0..1) |
| --page-scale-overlay | _page_scale_type | 1.0 | Skala strony przy overlay |
| --bg-vignette | _parallax_type | 0.15 | Siła winiety tła |
| --look | str | none | Preset koloru tła |
| --items-from | str | None | Folder z maskami paneli |