# Environment

The project relies on external binaries for rendering and OCR. Discovery is handled by `ken_burns_reel/bin_config.py` which searches CLI flags, environment variables and finally `$PATH`.

| Binary | Resolution order |
|--------|-----------------|
| ImageMagick `magick` | `--magick` → `IMAGEMAGICK_BINARY` → `shutil.which("magick")` |
| Tesseract OCR | `--tesseract` → `TESSERACT_BINARY` → `pytesseract.pytesseract.tesseract_cmd` → `shutil.which("tesseract")` |
| FFmpeg | probed via `shutil.which("ffmpeg")` in tools/diagnose_env.py |

Overrides are stored back in environment variables so subsequent calls reuse the resolved paths.

A helper script `tools/diagnose_env.py` prints detected paths and versions:

```bash
python tools/diagnose_env.py
```

## Ubuntu setup

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr imagemagick ffmpeg libgl1
```

The `libgl1` package is required by OpenCV to avoid `ImportError: libGL.so.1`.
