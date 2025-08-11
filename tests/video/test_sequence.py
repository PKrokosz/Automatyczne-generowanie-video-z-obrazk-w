import numpy as np
from PIL import Image

from ken_burns_reel.builder import make_panels_cam_sequence


def make_img(path):
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def test_align_beat_snaps_start(tmp_path):
    img1 = tmp_path / 'a.png'
    img2 = tmp_path / 'b.png'
    make_img(img1)
    make_img(img2)
    beats = [0.0, 1.05, 2.0]
    clip = make_panels_cam_sequence(
        [str(img1), str(img2)],
        target_size=(64, 64),
        fps=30,
        xfade=0.0,
        align_beat=True,
        beat_times=beats,
    )
    clip_no = make_panels_cam_sequence(
        [str(img1), str(img2)],
        target_size=(64, 64),
        fps=30,
        xfade=0.0,
        align_beat=False,
    )
    start_aligned = clip.clips[1].start
    start_plain = clip_no.clips[1].start
    diff_aligned = abs(start_aligned - beats[1])
    diff_plain = abs(start_plain - beats[1])
    assert diff_aligned <= 0.08
    assert diff_aligned < diff_plain
