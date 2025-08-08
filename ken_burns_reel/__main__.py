"""Command line interface for ken_burns_reel."""
from __future__ import annotations

import argparse
import os

import moviepy.config as mpyconf
import pytesseract

from .builder import make_filmstrip
from .ocr import verify_tesseract_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Ken Burns style video")
    parser.add_argument("folder", help="Input folder with images and audio")
    parser.add_argument("--tesseract", help="Path to Tesseract binary")
    parser.add_argument("--magick", help="Path to ImageMagick binary")
    args = parser.parse_args()

    if args.magick:
        os.environ["IMAGEMAGICK_BINARY"] = args.magick
        mpyconf.change_settings({"IMAGEMAGICK_BINARY": args.magick})
    if args.tesseract:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract

    verify_tesseract_available()
    make_filmstrip(args.folder)


if __name__ == "__main__":
    main()
