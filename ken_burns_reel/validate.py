"""Argument validation helpers for ken_burns_reel CLI."""
from __future__ import annotations

from argparse import Namespace
from typing import List

MIN_READ = 1.4


def validate_args(args: Namespace) -> List[str]:
    """Validate parsed CLI arguments.

    Returns a list of human readable error messages. The caller should abort
    if the list is non-empty.
    """
    min_read = max(MIN_READ, args.readability_ms / 1000.0)
    errors: List[str] = []
    if args.dwell < min_read:
        errors.append(f"--dwell {args.dwell:.2f}s < min_read {min_read:.2f}s")
    if args.min_dwell < min_read:
        errors.append(f"--min-dwell {args.min_dwell:.2f}s < min_read {min_read:.2f}s")
    if args.bpm:
        beat = 60.0 / args.bpm
        if args.beats_per_panel * beat < min_read:
            errors.append(
                "--beats-per-panel with --bpm shortens dwell below min_read"
            )
    return errors
