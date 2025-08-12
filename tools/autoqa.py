"""Generate per-file overrides from autovideo diagnostics CSV.

This script reads ``autovideo_diagnostics.csv`` and emits a YAML file
(``autovideo_presets_fix.yaml``) with overrides to be applied during a
subsequent render pass.  Each row in the CSV maps to a file and contains
quality diagnostics that decide the overrides to write.

The following columns are recognised:

``file`` (or ``filename``/``path``)
    Identifier of the asset.

``motion_score``
    Floating point score describing motion; low scores trigger stronger
    movement, high scores reduce it.

``letterbox_flag``
    Boolean flag indicating the asset requires a letterbox.

``dark_flag``
    Boolean flag; when set the background is brightened and the foreground
    glow increased.

``length_outlier``
    Boolean flag marking clips that were significantly longer/shorter than
    expected; forces per-panel dwell values.

The generated YAML maps file identifiers to dictionaries of CLI overrides.
Numeric adjustments are expressed as floats; consumers are expected to apply
them additively to their defaults.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Any

import yaml


def _truthy(val: str | None) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes"}


def generate_fix(csv_path: Path, yaml_path: Path) -> Dict[str, Dict[str, Any]]:
    """Read *csv_path* and write overrides to *yaml_path*.

    Returns the generated mapping for convenience/testing.
    """

    with csv_path.open(newline="", encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))

    fixes: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        name = row.get("file") or row.get("filename") or row.get("path")
        if not name:
            continue
        cfg: Dict[str, Any] = {}

        try:
            motion_score = float(row.get("motion_score", ""))
        except ValueError:
            motion_score = 0.0

        if motion_score and motion_score < 0.4:
            cfg.update({"travel": 0.3, "zoom-max": 0.08, "page-scale": -0.02, "dwell": -0.2})
        elif motion_score and motion_score > 0.6:
            cfg.update({"travel": -0.3, "transition-duration": 0.1, "page-scale": 0.02})

        if _truthy(row.get("letterbox_flag")):
            cfg["letterbox"] = True
            cfg["page-scale"] = cfg.get("page-scale", 0.0) - 0.03

        if _truthy(row.get("dark_flag")):
            cfg["bg-tone-strength"] = 0.7
            cfg["fg-glow"] = 0.08

        if _truthy(row.get("length_outlier")):
            cfg["min-dwell"] = 1.4
            cfg["dwell-mode"] = "each"

        if cfg:
            fixes[name] = cfg

    with yaml_path.open("w", encoding="utf8") as fh:
        yaml.safe_dump(fixes, fh, sort_keys=True)

    return fixes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="autovideo_diagnostics.csv",
        help="Path to diagnostics CSV",
    )
    parser.add_argument(
        "-o",
        "--out",
        default="autovideo_presets_fix.yaml",
        help="Output YAML path",
    )
    args = parser.parse_args(argv)

    generate_fix(Path(args.csv_path), Path(args.out))


if __name__ == "__main__":  # pragma: no cover
    main()

