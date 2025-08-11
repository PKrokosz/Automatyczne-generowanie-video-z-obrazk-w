# Code Quality

## Ruff

`ruff check .` reported 144 issues, mainly `E501` line length violations in tests such as `tests/video/test_new_features.py` and `tools/diagnose_env.py`.

## Mypy

`mypy .` reported 57 type errors across modules like `ken_burns_reel/builder.py` and `ken_burns_reel/__main__.py`.

These checks run in CI but do not block development yet.
