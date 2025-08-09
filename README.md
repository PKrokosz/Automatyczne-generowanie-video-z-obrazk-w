# Automatyczne Generowanie Wideo z Obrazków

Skrypt w Pythonie do automatycznego tworzenia wideo z serii obrazków, wykorzystujący efekt **Ken Burns** (powiększanie/przesuwanie), możliwość przewijania w osi X/Y, dodawania dźwięku, przejść i napisów z OCR.

Projekt zawiera modularną architekturę z możliwością modyfikacji poszczególnych etapów — od wykrywania punktu ostrości po eksport finalnego wideo.

---

## 📂 Struktura projektu

```
ken_burns_scroll_audio.py      # Skrypt CLI do generowania wideo ze skrolowaniem i audio
moviepy_config_defaults.py     # Domyślne ustawienia MoviePy
ken_burns_reel/
 ├── __main__.py                # Główny punkt wejścia pakietu
 ├── __init__.py
 ├── audio.py                   # Obsługa dźwięku w klipach
 ├── builder.py                 # Budowanie sekwencji wideo z efektami
 ├── config.py                  # Konfiguracja projektu
 ├── focus.py                   # Wykrywanie punktu ostrości na obrazie
 ├── ocr.py                     # OCR z użyciem Tesseract
 ├── transitions.py             # Efekty przejść między klipami
 ├── utils.py                   # Funkcje pomocnicze
tests/
 ├── test_audio.py              # Testy modułu audio
 ├── test_focus.py              # Testy wykrywania punktu ostrości
 ├── test_ocr.py                # Testy OCR
docs/
 └── AUDIT.md                   # Dokument audytu kodu
```

---

## 🚀 Instalacja

1. **Klonowanie repozytorium**
```bash
git clone https://github.com/<twoje-repo>/Automatyczne-generowanie-video-z-obrazkow.git
cd Automatyczne-generowanie-video-z-obrazkow
```

2. **Instalacja zależności**
```bash
pip install -r requirements.txt
```

3. **Wymagania dodatkowe**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — wymagany do ekstrakcji tekstu z obrazów (OCR)
- FFmpeg — wymagany przez MoviePy do renderowania wideo

### Konfiguracja binarek

Ścieżki do narzędzi zewnętrznych są rozwiązywane automatycznie w kolejności:

1. Parametry CLI `--magick` / `--tesseract`.
2. Zmienne środowiskowe `IMAGEMAGICK_BINARY` / `TESSERACT_BINARY`.
3. Ustawienia bibliotek (MoviePy, pytesseract).
4. Wyszukanie w `PATH` systemowym.

Jeśli narzędzie nie zostanie znalezione, napisy/OCR mogą zostać pominięte.
Na Windows upewnij się, że:

- katalog ImageMagick zawiera poprawny plik `colors.xml`,
- `tesseract.exe` znajduje się w `PATH` lub podaj do niego pełną ścieżkę.

---

## 📖 Użycie

### 1. Uruchomienie pakietu jako modułu
```bash
python -m ken_burns_reel <ścieżka_do_folderu_z_obrazkami> --output output.mp4
```

### 2. Skrypt przewijania + audio
```bash
python ken_burns_scroll_audio.py --input obrazy/ --audio muzyka.mp3 --output wideo.mp4
```

**Najważniejsze opcje CLI**:
- `--input` — katalog z obrazami
- `--audio` — ścieżka do pliku audio
- `--output` — nazwa pliku wynikowego
- `--duration` — czas trwania wideo w sekundach
- `--zoom` — poziom powiększenia efektu Ken Burns
- `--scroll` — włączenie przewijania obrazu

---

## 🔧 Moduły

### `focus.py`
- `detect_focus_point(image)` — wykrywa główny punkt ostrości obrazu na podstawie jasności i kontrastu.

### `ocr.py`
- `verify_tesseract_available()` — sprawdza, czy Tesseract jest zainstalowany.
- `extract_text_from_image(image)` — odczytuje tekst z obrazu.

### `audio.py`
- `add_audio_to_clip(clip, audio_path)` — dodaje ścieżkę audio do klipu.
- `fit_audio_duration(clip, audio_path)` — dopasowuje długość audio do klipu.

### `builder.py`
- Funkcje do tworzenia sekwencji klipów z efektami Ken Burns, przesunięciem, zoomem i OCR.

### `transitions.py`
- `crossfade_clips(clips, duration)` — dodaje efekt przenikania między klipami.
- Inne efekty przejść.

### `utils.py`
- Funkcje wspólne dla wielu modułów (np. ładowanie obrazów, walidacja plików).

---

## 🧪 Testy
Aby uruchomić testy:
```bash
pytest tests/
```

---

## 📜 Licencja
Projekt na licencji MIT — patrz [LICENSE](LICENSE).

---

## ✨ Autorzy
Projekt stworzony przez **[Twoje Imię / Nick]**.
