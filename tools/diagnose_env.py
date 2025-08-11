import shutil
import subprocess
from pathlib import Path

BINARIES = {
    "tesseract": ["tesseract", "--version"],
    "imagemagick": ["magick", "-version"],
    "ffmpeg": ["ffmpeg", "-version"],
}


def check_bin(name: str, cmd: list[str]) -> None:
    path = shutil.which(cmd[0])
    if not path:
        print(f"{name}: NOT FOUND")
        return
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).splitlines()[0]
    except Exception as e:  # noqa: BLE001
        out = f"error invoking: {e}"
    print(f"{name}: {path} -> {out}")


def main() -> None:
    for name, cmd in BINARIES.items():
        check_bin(name, cmd)
    try:
        libgl_path = next(Path("/usr/lib").rglob("libGL.so.1"))
        print(f"libGL: {libgl_path}")
    except StopIteration:
        print("libGL: NOT FOUND")


if __name__ == "__main__":
    main()
