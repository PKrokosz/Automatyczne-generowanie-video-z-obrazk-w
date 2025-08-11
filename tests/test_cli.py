import pytest
from ken_burns_reel import cli


def test_validate_blocks_negative_times(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--validate", "--trans-dur", "-1"])
    err = capsys.readouterr().err
    assert "--trans-dur" in err
