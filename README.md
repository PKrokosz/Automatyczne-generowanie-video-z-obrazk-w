# Automatyczne Generowanie Wideo z Obrazków

Skrypt w Pythonie do automatycznego tworzenia wideo z serii obrazków, wykorzystujący efekt **Ken Burns** (powiększanie/przesuwanie), możliwość przewijania w osi X/Y, dodawania dźwięku, przejść i napisów z OCR.

Projekt zawiera modularną architekturę z możliwością modyfikacji poszczególnych etapów — od wykrywania punktu ostrości po eksport finalnego wideo.

---

## 🚀 Quick start

**Bash**

```bash
python -m ken_burns_reel . --mode panels \
  --bg-mode blur --page-scale 0.94 --bg-parallax 0.85 \
  --profile social
```

**PowerShell** (multiline używa backticka \`)

```powershell
python -m ken_burns_reel . `
  --mode panels `
  --bg-mode blur `
  --page-scale 0.94 `
  --bg-parallax 0.85 `
  --profile social
```

**CMD** (multiline używa znaku ^)

```cmd
python -m ken_burns_reel . --mode panels ^
  --bg-mode blur ^
  --page-scale 0.94 ^
  --bg-parallax 0.85 ^
  --profile social
```

## One-click

Tryb eksperymentalny generujący film z folderu stron i pliku audio jedną komendą.

**PowerShell**

```powershell
python -m ken_burns_reel . `
  --oneclick `
  --limit-items 10 `
  --align-beat `
  --profile preview `
  --aspect 9:16 --height 1080
```

**Bash / CMD** (jedna linia):

```bash
python -m ken_burns_reel . --oneclick --limit-items 10 --align-beat --profile preview --aspect 9:16 --height 1080
```

Obrazy mogą znajdować się w bieżącym katalogu lub podfolderze `pages/`. Dźwięk szukany jest w katalogu nadrzędnym.

Przykłady wymiarowania:

```bash
# 16:9 poziomo
python -m ken_burns_reel . --mode panels --size 1920x1080 \
  --bg-mode blur --page-scale 0.94 --profile social

# 9:16 pionowo
python -m ken_burns_reel . --mode panels --aspect 9:16 --height 1080 \
  --bg-mode blur --page-scale 0.94 --profile social
```

Rekomendowana wartość `--page-scale` mieści się w zakresie `0.90–0.95`.

---

### Eksport paneli

```bash
python -m ken_burns_reel input_pages --export-panels panels --export-mode rect
```

### Tryb panel-first

**PowerShell** (multiline backtick)

```powershell
python -m ken_burns_reel .\panels `
  --mode panels-items `
  --trans smear --trans-dur 0.32 --smear-strength 1.1 `
  --bg-mode blur --page-scale 0.92 --bg-parallax 0.85 `
  --profile preview
```

**Bash**

```bash
python -m ken_burns_reel panels --mode panels-items \
  --size 1920x1080 --trans whip --trans-dur 0.28 \
  --profile social
```

### Overlay mode (page + masked panels)

```bash
python -m ken_burns_reel . --mode panels-overlay \
  --overlay-fit 0.75 --bg-source page \
  --parallax-bg 0.85 --parallax-fg 0.0
```

W tym trybie pełna strona stanowi tło z płynnym ruchem między panelami,
a pojedynczy panel (z zachowaną białą ramką) pojawia się na środku
kadru jako nakładka z cieniem.

`--bg-source`:

- `page` (crop strony z toningiem)
- `blur`
- `stretch`
- `gradient`

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
pip install -e .
```

3. **Wymagania dodatkowe**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — wymagany do ekstrakcji tekstu z obrazów (OCR)
- [ImageMagick](https://imagemagick.org) — renderowanie podpisów
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

**Najważniejsze opcje CLI**:
- `--audio-fit {trim,silence,loop}` — dopasowanie długości audio do wideo
- `--audio-fit trim` — jeśli audio jest krótsze niż wideo, narzędzie automatycznie dopełnia ciszą (z fade-in/out), żeby uniknąć błędów odtwarzacza.
- `--dwell-mode {first,each}` — zatrzymanie tylko na pierwszym panelu lub na każdym
- `--align-beat` — dociąga start stron do beatu (±0.08 s, bez ujemnych segmentów)
- `--debug-panels` — zapisuje podgląd wykrytych paneli i kończy działanie
- `--bg-mode {none,blur,stretch,gradient}` — sposób wypełnienia tła
- `--bg-parallax` — siła paralaksy tła (0–1)
- `--page-scale` — skala strony w kadrze (0.80–1.0)
- `--panel-bleed` — margines przy kadrowaniu panelu (px)
- `--zoom-max` — maksymalne dodatkowe przybliżenie małego tekstu
- `--travel-ease {in,out,inout,linear}` — easing ruchu kamery
- `--size WxH` lub `--aspect 9:16|16:9|1:1 --height H` — docelowy rozmiar wideo
- `--profile {preview,social,quality}` / `--preview` — presety eksportu (jakość vs szybkość)

Czas trwania filmu wynika z sumy klipów wideo, a audio jest dostosowywane zgodnie z `--audio-fit`.

---

### Formaty i presety

Przykłady użycia:

**Bash**
```bash
python -m ken_burns_reel folder --bg-mode blur --profile social
```

**CMD**
```cmd
python -m ken_burns_reel folder --bg-mode blur --profile social
```

**PowerShell** (multiline używa backticka `, a nie ^)
```powershell
python -m ken_burns_reel folder `
  --bg-mode blur `
  --page-scale 0.92
```

Zalecane `--page-scale` w zakresie `0.90–0.95`. Wideo można wymiarować przez `--size WxH` albo `--aspect 9:16 --height 1080`.

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
