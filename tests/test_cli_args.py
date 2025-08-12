import logging
import pytest
from ken_burns_reel import cli


def test_tight_border_parser():
    parser = cli.build_parser()
    args = parser.parse_args(["--tight-border", "10"])
    assert args.tight_border == 10
    assert cli.validate_args(args) == []


def test_tight_border_out_of_range():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tight-border", "200"])


def test_trans_dur_alias_warns(caplog):
    with caplog.at_level(logging.WARNING):
        cli.main(["--trans-dur", "0.5", "--validate"])
    assert "deprecated" in caplog.text
