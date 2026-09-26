from __future__ import annotations

import pytest

from wave_analysis.cli import _parse_years, build_parser, main


def test_parser_and_years():
    args = build_parser().parse_args(["ndbc", "download", "41010", "--years", "2020-2022,2025"])
    assert args.stations == ["41010"]
    assert _parse_years(args.years) == [2020, 2021, 2022, 2025]
    assert _parse_years(None) is None


def test_registry_validate_command(capsys):
    assert main(["registry", "validate"]) == 0
    assert "OK" in capsys.readouterr().out


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "wave-analysis" in capsys.readouterr().out
