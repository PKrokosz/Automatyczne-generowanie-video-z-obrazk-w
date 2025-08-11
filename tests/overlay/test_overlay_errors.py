import sys
import tempfile

import pytest
from PIL import Image

from ken_burns_reel import __main__ as kb_main


def test_temp_export_failure(monkeypatch, tmp_path):
    img = tmp_path / "page1.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(img)

    def raise_perm():
        raise PermissionError("no perm")

    monkeypatch.setattr(tempfile, "mkdtemp", raise_perm)
    monkeypatch.setattr(kb_main, "verify_tesseract_available", lambda: None)
    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path), "--mode", "panels-overlay"])

    with pytest.raises(SystemExit) as exc:
        kb_main.main()
    msg = str(exc.value).lower()
    assert "failed to export panels" in msg
