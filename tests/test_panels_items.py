import os
import numpy as np
from PIL import Image

from ken_burns_reel.panels import export_panels
from ken_burns_reel.transitions import (
    smear_transition,
    whip_pan_transition,
    smear_bg_crossfade_fg,
)
from ken_burns_reel.builder import make_panels_items_sequence, make_panels_overlay_sequence

try:
    from moviepy.editor import ColorClip
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy import ColorClip


def _make_test_page(tmp_path):
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    # two dark panels
    import cv2

    cv2.rectangle(arr, (10, 10), (90, 90), (0, 0, 0), -1)
    cv2.rectangle(arr, (110, 10), (190, 90), (0, 0, 0), -1)
    p = tmp_path / "page.png"
    Image.fromarray(arr).save(p)
    return p


def test_export_panels_rect_and_mask(tmp_path):
    page = _make_test_page(tmp_path)
    out_rect = tmp_path / "rect"
    paths = export_panels(str(page), str(out_rect), mode="rect")
    assert len(paths) == 2
    assert all(os.path.getsize(p) > 0 for p in paths)

    out_mask = tmp_path / "mask"
    paths_m = export_panels(str(page), str(out_mask), mode="mask")
    assert len(paths_m) == 2
    with Image.open(paths_m[0]) as im:
        assert im.mode == "RGBA"
        alpha = np.array(im)[:, :, 3]
        assert np.any(alpha < 255)


def test_smear_transition_basic():
    clip1 = ColorClip(size=(100, 80), color=(255, 0, 0)).set_duration(1).set_fps(24)
    clip2 = ColorClip(size=(100, 80), color=(0, 255, 0)).set_duration(1).set_fps(24)
    trans = smear_transition(clip1, clip2, 0.2, (100, 80), vec=(20, 0))
    f0 = trans.get_frame(0)
    fmid = trans.get_frame(0.1)
    fend = trans.get_frame(0.2 - 1e-6)
    assert np.allclose(f0, clip1.get_frame(0.8), atol=1)
    assert np.allclose(fend, clip2.get_frame(0), atol=1)
    assert not np.allclose(fmid, f0)
    assert not np.allclose(fmid, fend)


def test_whip_pan_transition_basic():
    clip1 = ColorClip(size=(100, 80), color=(0, 0, 255)).set_duration(1).set_fps(24)
    clip2 = ColorClip(size=(100, 80), color=(0, 255, 0)).set_duration(1).set_fps(24)
    trans = whip_pan_transition(
        clip1, clip2, 0.2, (100, 80), vec=(20, 0), ease="linear"
    )
    f0 = trans.get_frame(0)
    fend = trans.get_frame(0.2 - 1e-6)
    assert np.allclose(f0, clip1.get_frame(0.8), atol=1)
    assert np.allclose(fend, clip2.get_frame(0), atol=1)


def test_make_panels_items_sequence_duration(tmp_path):
    p1 = tmp_path / "p1.png"
    p2 = tmp_path / "p2.png"
    Image.new("RGB", (50, 50), (10, 20, 30)).save(p1)
    Image.new("RGB", (50, 50), (30, 20, 10)).save(p2)
    clip = make_panels_items_sequence([str(p1), str(p2)], dwell=0.5, trans="smear", trans_dur=0.3)
    assert abs(clip.duration - (0.5 * 2 + 0.3)) < 0.05
    assert clip.size == [1080, 1920] or clip.size == (1080, 1920)


def test_overlay_center_and_fit(tmp_path):
    page = _make_test_page(tmp_path)
    mask_dir = tmp_path / "mask" / "page_0001"
    paths = export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(200, 100),
        dwell=0.1,
        travel=0.1,
        overlay_fit=0.75,
    )
    m = clip.mask.get_frame(0)
    ys, xs = np.where(m > 0.1)
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    assert abs(cx - clip.w / 2) <= 1
    assert abs(cy - clip.h / 2) <= 1
    h = ys.max() - ys.min()
    assert abs(h - 0.75 * clip.h) / clip.h < 0.02


def test_overlay_background_moves(tmp_path):
    page = _make_test_page(tmp_path)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(200, 100),
        dwell=0.1,
        travel=0.2,
    )
    f1 = clip.get_frame(0.11)
    f2 = clip.get_frame(0.27)
    assert not np.allclose(f1, f2)
    assert f1.mean() > 1


def test_smear_bg_crossfade_fg_edges():
    from moviepy.editor import ColorClip, ImageClip
    import cv2
    import numpy as np

    bg1 = ColorClip(size=(50, 50), color=(255, 0, 0)).set_duration(1).set_fps(24)
    bg2 = ColorClip(size=(50, 50), color=(0, 255, 0)).set_duration(1).set_fps(24)
    arr = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.rectangle(arr, (5, 5), (45, 45), (255, 255, 255, 255), 2)
    fg = ImageClip(arr[:, :, :3], ismask=False).set_mask(
        ImageClip(arr[:, :, 3] / 255.0, ismask=True)
    ).set_duration(1).set_fps(24)
    trans = smear_bg_crossfade_fg(
        bg1,
        bg2,
        fg,
        fg,
        0.3,
        (50, 50),
        vec=(20, 0),
        fps=24,
        bg_brightness_dip=0.05,
        steps_auto=True,
    )
    frame_start = trans.get_frame(0)
    frame_mid = trans.get_frame(0.15)
    edges = cv2.Canny(frame_mid, 50, 150)
    assert edges.mean() > 5
    assert frame_mid.mean() < frame_start.mean()
