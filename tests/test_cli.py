from ken_burns_reel.__main__ import parse_args


def test_preset_overrides_flags(tmp_path):
    preset = tmp_path / "preset.yaml"
    preset.write_text("dwell: 2.5\n")

    args = parse_args([str(tmp_path), "--preset", str(preset)])
    assert args.dwell == 2.5

    args_cli = parse_args(
        [str(tmp_path), "--preset", str(preset), "--dwell", "1.5"]
    )
    assert args_cli.dwell == 1.5
