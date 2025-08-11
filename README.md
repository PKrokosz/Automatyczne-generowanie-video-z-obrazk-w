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

## Kolor
Ścieżka przetwarzania zachowuje stałą konwersję sRGB → linear → sRGB przy
użyciu 16‑bitowej precyzji.  Pomaga to zminimalizować dryf barw i zapewnia
powtarzalne wyniki.  Funkcje pomocnicze znajdują się w module
`ken_burns_reel.color` i są objęte testem "color‑lock" porównującym
histogramy kanałów.

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

Szczegółowe przykłady CLI znajdują się w [docs/cli_examples.md](docs/cli_examples.md).

## Tests
```bash
pytest tests/
```

## Contributing
Zasady współpracy opisuje [CONTRIBUTING.md](CONTRIBUTING.md).

## License
Projekt na licencji MIT — patrz [LICENSE](LICENSE).
