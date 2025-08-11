import numpy as np

try:
    from moviepy import ColorClip, CompositeVideoClip
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    from moviepy.editor import ColorClip, CompositeVideoClip

from ken_burns_reel.transitions import fg_fade, _get_ease_fn


def _set_duration(clip, duration):
    return getattr(clip, "with_duration", clip.set_duration)(duration)


def _set_fps(clip, fps):
    return getattr(clip, "with_fps", clip.set_fps)(fps)


def _set_opacity(clip, opacity):
    return getattr(clip, "with_opacity", clip.set_opacity)(opacity)


def test_fg_fade_keeps_background():
    bg = _set_fps(_set_duration(ColorClip(size=(4, 4), color=(0, 0, 0)), 1), 1)
    fg = _set_opacity(
        _set_fps(_set_duration(ColorClip(size=(4, 4), color=(255, 0, 0)), 1), 1),
        1,
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
