
from ken_burns_reel.__main__ import parse_args
import pytest
from ken_burns_reel import cli

def test_preset_overrides_flags(tmp_path):
    preset = tmp_path / "preset.yaml"
    preset.write_text("dwell: 2.5\n")

    args = parse_args([str(tmp_path), "--preset", str(preset)])
    assert args.dwell == 2.5

    args_cli = parse_args(
        [str(tmp_path), "--preset", str(preset), "--dwell", "1.5"]
    )
    assert args_cli.dwell == 1.5


def test_validate_blocks_negative_times(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--validate", "--trans-dur", "-1"])
    err = capsys.readouterr().err
    assert "--trans-dur" in err

