from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import List

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ken Burns Reel helper CLI")
    parser.add_argument("--trans", default="none", help="Transition type")
    parser.add_argument(
        "--transition-duration",
        dest="trans_dur",
        type=float,
        default=0.0,
        help="Transition duration in seconds",
    )
    parser.add_argument(
        "--trans-dur",
        dest="trans_dur",
        type=float,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fg-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--validate", action="store_true", help="Validate arguments and exit"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Force deterministic build"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--bg-offset", type=float, default=0.0)
    parser.add_argument("--fg-offset", type=float, default=0.0)
    parser.add_argument("--min-read", type=float, default=1.4)
    parser.add_argument("--drift-x", type=float, default=0.0)
    parser.add_argument("--drift-y", type=float, default=0.0)
    parser.add_argument("--rot-deg", type=float, default=0.0)
    parser.add_argument("--parallax", type=float, default=0.0)
    parser.add_argument(
        "--detect-bubbles",
        choices=["on", "off"],
        default="off",
        help="Enable speech bubble detection",
    )
    parser.add_argument(
        "--bubble-mode",
        choices=["mask", "rect"],
        default="mask",
    )
    parser.add_argument("--bubble-min-area", type=int, default=200)
    parser.add_argument("--bubble-roundness-min", type=float, default=0.3)
    parser.add_argument("--bubble-contrast-min", type=float, default=0.0)
    parser.add_argument(
        "--bubble-export",
        choices=["none", "masks", "keyframes"],
        default="none",
    )
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if args.trans_dur < 0:
        errors.append("--transition-duration must be >= 0")
    if args.bg_offset and args.fg_offset:
        errors.append("conflict: --bg-offset with --fg-offset")
    if args.min_read < 1.4:
        errors.append("--min-read must be >= 1.4")
    if abs(args.drift_x) > 0.01:
        errors.append("--drift-x out of range")
    if abs(args.drift_y) > 0.01:
        errors.append("--drift-y out of range")
    if abs(args.rot_deg) > 0.3:
        errors.append("--rot-deg out of range")
    if not (0.0 <= args.parallax <= 0.06):
        errors.append("--parallax out of range")
    if args.detect_bubbles == "on" and args.bubble_min_area <= 0:
        errors.append("--bubble-min-area must be > 0")
    if not (0.0 <= args.bubble_roundness_min <= 1.0):
        errors.append("--bubble-roundness-min must be within [0,1]")
    if args.bubble_contrast_min < 0.0:
        errors.append("--bubble-contrast-min must be >= 0")
    return errors


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fg_only:
        logging.warning("--fg-only is deprecated; use --trans fg-fade")
        args.trans = "fg-fade"

    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logging.info("deterministic build seed=%s", args.seed)

    if args.validate:
        errs = validate_args(args)
        if errs:
            for e in errs:
                print(f"validation error: {e}", file=sys.stderr)
            raise SystemExit(1)
        return

    # Placeholder for future CLI actions
    logging.info("CLI completed")


if __name__ == "__main__":  # pragma: no cover
    main()
