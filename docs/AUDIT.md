# AUDYT funkcji i zależności

## Przegląd modułów

### `ken_burns_reel/audio.py`
- `extract_beats(audio_path: str) -> List[float]` – korzysta z `librosa` do wykrywania uderzeń w pliku audio.

### `ken_burns_reel/builder.py`
- `ken_burns_scroll(...)` – tworzy pojedynczy klip w stylu Ken Burnsa; używa `moviepy` i `overlay_caption`.
- `make_filmstrip(input_folder: str) -> str` – główna orkiestracja: wyszukuje obrazy, dźwięk i składa finalne wideo.
  Wywołuje `extract_beats`, `extract_caption`, `detect_focus_point`, `overlay_caption`.

### `ken_burns_reel/config.py`
- Konfiguracja środowiska (`IMAGEMAGICK_BINARY`, `tesseract_cmd`). Definiuje rozszerzenia `IMAGE_EXTS`, `AUDIO_EXTS`.

### `ken_burns_reel/focus.py`
- `detect_focus_point(img: Image.Image) -> Tuple[int, int]` – detekcja twarzy (`opencv`) lub centroid jasności (`numpy`).

### `ken_burns_reel/ocr.py`
- `extract_caption(img_path: str) -> str` – OCR z użyciem `pytesseract`.
- `verify_tesseract_available() -> None` – sprawdza obecność binarki tesseract.

### `ken_burns_reel/transitions.py`
- `slide_transition(prev_clip, next_clip, duration, size, fps)` – prosty efekt przesuwania między klipami.

### `ken_burns_reel/utils.py`
- `overlay_caption(clip, text, size)` – nakłada napis na klip (`moviepy`).
- `smart_crop(img, target_w, target_h)` – kadrowanie obrazka do docelowych proporcji (`Pillow`).

### `ken_burns_reel/__main__.py`
- `main()` – CLI; sprawdza tesseract i uruchamia `make_filmstrip`.

## Grupy funkcjonalne
- **OCR/Caption**: `extract_caption`, `overlay_caption`, `verify_tesseract_available`.
- **Focus/Compose**: `detect_focus_point`, `smart_crop`, `ken_burns_scroll`.
- **Transitions**: `slide_transition`.
- **Audio/Sync**: `extract_beats`.
- **Orkiestracja**: `make_filmstrip`, wywoływana z `__main__.py`.

## Sprzężenia między modułami
- `builder.make_filmstrip` → `audio.extract_beats`, `ocr.extract_caption`, `focus.detect_focus_point`, `utils.overlay_caption`.
- `builder.ken_burns_scroll` → `utils.overlay_caption`.
- `__main__.py` → `ocr.verify_tesseract_available` + `builder.make_filmstrip`.

## Propozycja warstw pakietu
1. **Warstwa infrastruktury**: `config` (ustawienia zewnętrznych narzędzi).
2. **Warstwa przetwarzania danych**: `ocr`, `focus`, `audio`, `utils` (obróbka obrazów/tekstów/dźwięku).
3. **Warstwa kompozycji**: `builder` (łączenie klipów, synchronizacja) + `transitions`.
4. **Warstwa interfejsu**: `__main__` (CLI) oraz przyszły moduł `diagnostics` do logowania.

Struktura ta odpowiada schematowi z EPIC 2 i pozwala na dalszą modularizację (np. wydzielenie `image_utils.py`, `audio_sync.py`, `diagnostics.py`).
