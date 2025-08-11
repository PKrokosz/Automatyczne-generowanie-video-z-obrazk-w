import numpy as np

try:
    from moviepy.editor import VideoClip, CompositeVideoClip
except ModuleNotFoundError:  # moviepy >=2
    from moviepy import VideoClip, CompositeVideoClip

from ken_burns_reel.transitions import overlay_lift


def test_overlay_lift_bg_stable_and_shadow() -> None:
    panel = np.zeros((20, 20, 4), dtype=np.uint8)
    panel[2:18, 2:18, :3] = [200, 0, 0]
    panel[2:18, 2:18, 3] = 255
    clip = overlay_lift(panel, duration=0.3, fps=10)
    frame_start = clip.get_frame(0.0)
    alpha_start = clip.mask.get_frame(0.0)
    frame_end = clip.get_frame(0.3)
    alpha_end = clip.mask.get_frame(0.3)
    assert np.all(frame_start == 0)
    assert np.all(alpha_start == 0)
    border_mask = np.ones((20, 20), dtype=bool)
    border_mask[2:18, 2:18] = False
    assert np.any(frame_end[border_mask] > 0)
    assert np.count_nonzero(frame_end[..., 0] == 200) > 0
    reds = frame_end[frame_end[..., 0] == 200]
    assert np.all(reds[:, 1:] == 0)
    assert alpha_end.max() > 0.99
