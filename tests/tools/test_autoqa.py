from pathlib import Path

import yaml

from tools.autoqa import generate_fix


def write_csv(tmp_path: Path, rows: list[str]) -> Path:
    csv_path = tmp_path / "autovideo_diagnostics.csv"
    csv_path.write_text("file,motion_score,letterbox_flag,dark_flag,length_outlier\n" + "\n".join(rows))
    return csv_path


def test_generate_fix(tmp_path):
    csv_path = write_csv(
        tmp_path,
        [
            "a.jpg,0.2,1,0,0",
            "b.jpg,0.9,0,1,1",
            "c.jpg,0.5,0,0,0",
        ],
    )
    yaml_path = tmp_path / "fix.yaml"
    mapping = generate_fix(csv_path, yaml_path)

    with yaml_path.open() as fh:
        data = yaml.safe_load(fh)

    assert mapping == data
    assert "a.jpg" in data
    assert data["a.jpg"]["travel"] == 0.3
    assert data["a.jpg"]["letterbox"] is True
    assert data["b.jpg"]["transition-duration"] == 0.1
    assert data["b.jpg"]["bg-tone-strength"] == 0.7
    assert "c.jpg" not in data

