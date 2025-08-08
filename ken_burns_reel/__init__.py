"""Ken Burns reel package."""

__all__ = ["make_filmstrip"]


def make_filmstrip(*args, **kwargs):
    from .builder import make_filmstrip as _make_filmstrip

    return _make_filmstrip(*args, **kwargs)
