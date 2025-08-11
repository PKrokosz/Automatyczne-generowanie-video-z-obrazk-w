# Automatyczne Generowanie Wideo z Obrazków

Skrypt w Pythonie do automatycznego tworzenia wideo z serii obrazków, wykorzystujący efekt **Ken Burns**, przewijanie w osi X/Y, dodawanie dźwięku, przejścia oraz napisy z OCR.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [Contributing](#contributing)
- [License](#license)

## Features
- Efekt Ken Burns (zoom i przesunięcie) dla pojedynczych paneli.
- Obsługa przewijania w osi X/Y z paralaksą tła.
- Integracja z audio, przejściami i napisami generowanymi przez OCR.
- Modularna architektura umożliwiająca modyfikację etapów przetwarzania.
- Efekty warstwowe takie jak `page_shadow` oraz przejście `overlay_lift` z
  animowaną podmianą panelu.
- Możliwość ładowania presetów stylu z plików YAML (np. `styles/float_black_v1.yaml`).

## Installation
1. Zainstaluj zależności Pythona:
   ```bash
   pip install -e .
   ```
2. Zainstaluj dodatkowe narzędzia:
   - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
   - [ImageMagick](https://imagemagick.org)
   - FFmpeg (wymagany przez MoviePy)
3. **Konfiguracja binarek** – ścieżki są wykrywane w kolejności:
   1. Parametry CLI `--magick` / `--tesseract`
   2. Zmienne środowiskowe `IMAGEMAGICK_BINARY` / `TESSERACT_BINARY`
   3. Dla Tesseract: `pytesseract.pytesseract.tesseract_cmd`
   4. Wyszukanie w systemowym `PATH`
   Jeśli narzędzie nie zostanie znalezione, napisy/OCR mogą zostać pominięte.
   Na Windows upewnij się, że katalog ImageMagick zawiera `colors.xml`, a `tesseract.exe` znajduje się w `PATH`.

## Usage
```bash
python -m ken_burns_reel . --mode panels \
  --bg-mode blur --page-scale 0.94 --bg-parallax 0.85 \
  --profile social
```

Presety można wczytać przez `--preset <plik.yaml>`; repo zawiera m.in.
`styles/float_black_v1.yaml`. Hierarchia: domyślne < style preset < motion preset <
flagi CLI.

### 2-page test

**Bash**

```bash
python -m ken_burns_reel two_pages \
  --preset styles/float_black_v1.yaml \
  --transition-duration 0.3
```

**PowerShell**

```powershell
python -m ken_burns_reel two_pages `
  --preset styles/float_black_v1.yaml `
  --transition-duration 0.3
```

**CMD**

```cmd
python -m ken_burns_reel two_pages ^
  --preset styles\float_black_v1.yaml ^
  --transition-duration 0.3
```

Szczegółowe przykłady CLI znajdują się w [docs/cli_examples.md](docs/cli_examples.md).

One-liner (PowerShell/Bash/CMD):

```
python -m ken_burns_reel.cli --trans fg-fade --transition-duration 0.3 input_folder
```

## Tests
```bash
pytest tests/
```

## Contributing
Zasady współpracy opisuje [CONTRIBUTING.md](CONTRIBUTING.md).

## License
Projekt na licencji MIT — patrz [LICENSE](LICENSE).
