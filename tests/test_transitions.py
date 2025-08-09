from ken_burns_reel.transitions import slide_transition
from moviepy.editor import ColorClip


def test_slide_transition_basic():
    clip1 = ColorClip(size=(100, 100), color=(255, 0, 0)).set_duration(1).set_fps(24)
    clip2 = ColorClip(size=(100, 100), color=(0, 255, 0)).set_duration(1).set_fps(24)
    trans = slide_transition(clip1, clip2, 0.5, (100, 100), 24)
    assert abs(trans.duration - 0.5) < 1e-6
    frame = trans.get_frame(0.25)
    assert frame.shape[:2] == (100, 100)
