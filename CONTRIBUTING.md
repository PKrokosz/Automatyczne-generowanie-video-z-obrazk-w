# Współtworzenie projektu

Dziękujemy za chęć pomocy przy rozwoju repozytorium. Zanim rozpoczniesz pracę, przeczytaj proszę [AGENTS.md](agents.md), który zawiera szczegółowe wytyczne dotyczące środowiska, stylu kodu oraz uruchamiania testów.

## Instalacja środowiska

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .[dev]  # lub: pip install -r requirements.txt -r requirements-dev.txt
```

Wymagane są również zewnętrzne narzędzia:
- [ImageMagick](https://imagemagick.org) (`magick`)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`tesseract`)
- [FFmpeg](https://ffmpeg.org)

## Styl kodu i narzędzia

- Stosuj zasady PEP 8 i dodawaj adnotacje typów.
- Przed wysłaniem zmian uruchom linters:
  ```bash
  ruff check .
  mypy .
  ```
- Staraj się pisać małe, logiczne commity.

## Testy

Przed otwarciem PR uruchom wszystkie testy i podstawowe testy dymne:

```bash
pytest -q
python -m ken_burns_reel . --dry-run
```

## Workflow Pull Requestów

1. Utwórz fork i nowy branch dla swojej zmiany.
2. Zadbaj, aby każdy commit spełniał zasady z [AGENTS.md](agents.md).
3. Dołącz w opisie PR fragment logu z uruchomionych testów oraz linters.
4. Po pozytywnej weryfikacji otwórz PR do gałęzi `main`.

Miłego kodowania!
