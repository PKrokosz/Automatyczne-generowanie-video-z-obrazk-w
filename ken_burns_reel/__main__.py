"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse

from .builder import make_filmstrip
from .ocr import verify_tesseract_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Ken Burns style video")
    parser.add_argument("folder", help="Input folder with images and audio")
    args = parser.parse_args()
    verify_tesseract_available()
    make_filmstrip(args.folder)


if __name__ == "__main__":
    main()
