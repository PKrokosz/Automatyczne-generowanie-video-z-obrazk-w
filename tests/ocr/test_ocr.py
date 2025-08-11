import shutil
import pytest
from PIL import Image, ImageDraw

from ken_burns_reel.ocr import extract_caption

if shutil.which("tesseract") is None:
    pytest.skip("tesseract not installed", allow_module_level=True)


def test_extract_caption(tmp_path):
    img = Image.new("RGB", (200, 60), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Hello", fill="black")
    img_path = tmp_path / "sample.png"
    img.save(img_path)
    text = extract_caption(str(img_path))
    assert "hello" in text.lower()
