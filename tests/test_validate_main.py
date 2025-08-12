import pytest

cv2 = pytest.importorskip("cv2")
from ken_burns_reel import __main__ as kb_main


def test_validate_rejects_short_dwell(tmp_path, capsys):
    with pytest.raises(SystemExit):
        kb_main.main([str(tmp_path), "--validate", "--dwell", "1.0"])
    err = capsys.readouterr().err
    assert "--dwell" in err


def test_validate_rejects_beats_per_panel(tmp_path, capsys):
    with pytest.raises(SystemExit):
        kb_main.main([
            str(tmp_path),
            "--validate",
            "--bpm",
            "240",
            "--beats-per-panel",
            "0.5",
        ])
    err = capsys.readouterr().err
    assert "--beats-per-panel" in err
