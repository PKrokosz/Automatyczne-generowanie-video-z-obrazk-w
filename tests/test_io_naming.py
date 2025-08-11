from types import SimpleNamespace
from pathlib import Path
from types import SimpleNamespace

from ken_burns_reel.__main__ import _resolve_out_path


def test_auto_naming(tmp_path: Path) -> None:
    args = SimpleNamespace(
        out_naming="auto",
        out_prefix="test-",
        output=str(tmp_path),
        folder=str(tmp_path),
        mode="classic",
    )
    path = _resolve_out_path(args, "final_video.mp4", str(tmp_path))
    assert path.startswith(str(tmp_path)), path
    assert Path(path).name.startswith("test-")


def test_keep_naming(tmp_path: Path) -> None:
    args = SimpleNamespace(
        out_naming="keep",
        out_prefix="",
        output=str(tmp_path / "out.mp4"),
        folder=str(tmp_path),
        mode="classic",
    )
    path = _resolve_out_path(args, "final_video.mp4", str(tmp_path))
    assert Path(path).name.startswith("out")
