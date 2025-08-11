import numpy as np

try:
    from moviepy import ColorClip, CompositeVideoClip
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    from moviepy.editor import ColorClip, CompositeVideoClip

from ken_burns_reel.transitions import fg_fade, _get_ease_fn


def test_fg_fade_keeps_background():
    bg = ColorClip(size=(4, 4), color=(0, 0, 0)).set_duration(1).set_fps(1)
    fg = (
        ColorClip(size=(4, 4), color=(255, 0, 0))
        .set_duration(1)
        .set_fps(1)
        .set_opacity(1)
    )
    faded = fg_fade(fg, 1)
    comp = CompositeVideoClip([bg, faded])
    end_frame = comp.get_frame(1)
    assert np.all(end_frame == np.array([0, 0, 0]))

    ease_fn = _get_ease_fn("inout")
    for t in [0.0, 0.5, 1.0]:
        mask = faded.mask.get_frame(t)
        expected = np.full_like(mask, 1 - ease_fn(t))
        assert np.allclose(mask, expected, atol=1e-2)
