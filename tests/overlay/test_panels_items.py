import os
import numpy as np
import pytest
from PIL import Image

from ken_burns_reel.panels import export_panels
from ken_burns_reel.transitions import (
    smear_transition,
    whip_pan_transition,
    smear_bg_crossfade_fg,
)
from ken_burns_reel.builder import (
    make_panels_items_sequence,
    make_panels_overlay_sequence,
    _fit_window_to_box,
)

try:
    from moviepy.editor import ColorClip
except ModuleNotFoundError:  # moviepy >=2.0
    from moviepy import ColorClip


def _make_test_page(tmp_path, cv2):
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    # two dark panels
    cv2.rectangle(arr, (10, 10), (90, 90), (0, 0, 0), -1)
    cv2.rectangle(arr, (110, 10), (190, 90), (0, 0, 0), -1)
    p = tmp_path / "page.png"
    Image.fromarray(arr).save(p)
    return p


def _make_page_for_anchored(tmp_path, cv2):
    arr = np.full((900, 600, 3), 200, dtype=np.uint8)
    cv2.rectangle(arr, (50, 50), (250, 350), (0, 0, 0), -1)
    cv2.rectangle(arr, (50, 50), (250, 350), (255, 255, 255), 3)
    cv2.rectangle(arr, (350, 450), (550, 850), (0, 0, 0), -1)
    cv2.rectangle(arr, (350, 450), (550, 850), (255, 255, 255), 3)
    p = tmp_path / "page.png"
    Image.fromarray(arr).save(p)
    return p


def test_export_panels_rect_and_mask(tmp_path):
    cv2 = pytest.importorskip("cv2")
    page = _make_test_page(tmp_path, cv2)
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


def test_overlay_anchored_projection(tmp_path):
    cv2 = pytest.importorskip("cv2")
    page = _make_page_for_anchored(tmp_path, cv2)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(300, 450),
        dwell=0.2,
        travel=0.2,
        overlay_mode="anchored",
        overlay_scale=1.0,
        parallax_fg=0.0,
    )
    t = 0.1
    frame = clip.get_frame(t)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert cnts
    M = cv2.moments(cnts[0])
    cx = M["m10"] / max(1, M["m00"])
    cy = M["m01"] / max(1, M["m00"])
    box = (50, 50, 200, 300)
    cx0, cy0, w0, h0 = _fit_window_to_box(600, 900, box, (300, 450))
    win_w = w0
    win_h = h0
    left0 = int(max(0, min(cx0 - win_w // 2, 600 - win_w)))
    top0 = int(max(0, min(cy0 - win_h // 2, 900 - win_h)))
    S = 300 / win_w
    exp_cx = (box[0] + box[2] / 2 - left0) / win_w * 300
    exp_cy = (box[1] + box[3] / 2 - top0) / win_h * 450
    assert abs(cx - exp_cx) < 6
    assert abs(cy - exp_cy) < 6


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


def test_export_panels_gutter_thicken(tmp_path):
    cv2 = pytest.importorskip("cv2")
    arr = np.zeros((80, 120, 3), np.uint8)
    cv2.line(arr, (60, 0), (60, 79), (255, 255, 255), 1)
    page = tmp_path / "page.png"
    Image.fromarray(arr).save(page)
    out0 = tmp_path / "mask0" / "page_0001"
    out1 = tmp_path / "mask1" / "page_0001"
    paths0 = export_panels(str(page), str(out0), mode="mask", bleed=0, tight_border=0, feather=0, gutter_thicken=0)
    paths1 = export_panels(str(page), str(out1), mode="mask", bleed=0, tight_border=0, feather=0, gutter_thicken=4)
    assert len(paths0) == 1
    assert len(paths1) >= 2


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
    cv2 = pytest.importorskip("cv2")
    page = _make_test_page(tmp_path, cv2)
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
    cv2 = pytest.importorskip("cv2")
    page = _make_test_page(tmp_path, cv2)
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
    cv2 = pytest.importorskip("cv2")
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


def test_overlay_travel_ease(tmp_path):
    cv2 = pytest.importorskip("cv2")
    page = _make_test_page(tmp_path, cv2)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    # make foreground transparent to examine background crop directly
    for img_path in mask_dir.glob("panel_*.png"):
        with Image.open(img_path).convert("RGBA") as im:
            arr = np.array(im)
        arr[:, :, 3] = 0
        Image.fromarray(arr).save(img_path)

    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(200, 100),
        dwell=0.1,
        travel=0.4,
        travel_ease="linear",
        parallax_bg=1.0,
        parallax_fg=0.0,
        overlay_fit=0.75,
    )
    mid_t = 0.1 + 0.4 * 0.5
    frame = clip.get_frame(mid_t)

    # expected crop center
    box0 = (10, 10, 80, 80)
    box1 = (110, 10, 80, 80)
    cx0, cy0, w0, h0 = _fit_window_to_box(200, 100, box0, (200, 100))
    cx1, cy1, w1, h1 = _fit_window_to_box(200, 100, box1, (200, 100))
    win_w = max(w0, w1)
    win_h = max(h0, h1)
    left0 = int(max(0, min(cx0 - win_w // 2, 200 - win_w)))
    left1 = int(max(0, min(cx1 - win_w // 2, 200 - win_w)))
    top0 = int(max(0, min(cy0 - win_h // 2, 100 - win_h)))
    top1 = int(max(0, min(cy1 - win_h // 2, 100 - win_h)))
    exp_left = left0 + (left1 - left0) * 0.5
    exp_top = top0 + (top1 - top0) * 0.5
    exp_cx = exp_left + win_w / 2
    exp_cy = exp_top + win_h / 2

    # locate actual crop within page via template matching
    page_arr = np.array(Image.open(page).convert("RGB"))
    frame_small = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(page_arr, frame_small, cv2.TM_SQDIFF)
    _, _, loc, _ = cv2.minMaxLoc(res)
    act_cx = loc[0] + win_w / 2
    act_cy = loc[1] + win_h / 2

    assert abs(act_cx - exp_cx) <= 12
    assert abs(act_cy - exp_cy) <= 12


def test_overlay_parallax_fg(tmp_path):
    cv2 = pytest.importorskip("cv2")
    page = _make_test_page(tmp_path, cv2)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(200, 100),
        dwell=0.1,
        travel=0.2,
        travel_ease="linear",
        parallax_bg=0.0,
        parallax_fg=0.1,
    )
    t_start = 0.1 + 0.01
    t_end = 0.1 + 0.2 - 0.01
    f0 = clip.get_frame(t_start)
    f1 = clip.get_frame(t_end)
    assert np.mean(np.abs(f1.astype(np.int16) - f0.astype(np.int16))) > 1


def test_overlay_enhance_applied(tmp_path):
    cv2 = pytest.importorskip("cv2")
    arr = np.full((100, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(arr, (10, 10), (90, 90), (200, 200, 200), -1)
    cv2.rectangle(arr, (10, 10), (90, 90), (0, 0, 0), 2)
    cv2.line(arr, (10, 50), (90, 50), (205, 205, 205), 1)
    page = tmp_path / "page.png"
    Image.fromarray(arr).save(page)
    mask_dir = tmp_path / "mask" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    panel_path = mask_dir / "panel_0001.png"
    with Image.open(panel_path) as im:
        orig = np.array(im.convert("RGB"))

    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "mask"),
        target_size=(200, 100),
        dwell=0.1,
        travel=0.1,
        parallax_bg=0.0,
        parallax_fg=0.0,
        overlay_fit=0.5,
        overlay_pop=1.25,
    )
    frame = clip.get_frame(0)
    mask = clip.mask.get_frame(0)
    ys, xs = np.where(mask > 0.1)
    fg = frame[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    def sobel_var(a: np.ndarray) -> float:
        g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
        mag2 = gx ** 2 + gy ** 2
        return float(mag2.var())

    assert sobel_var(fg) > sobel_var(orig)


def test_smear_bg_brightness_dip():
    from moviepy.editor import ColorClip

    bg1 = ColorClip(size=(50, 50), color=(50, 50, 50)).set_duration(1).set_fps(24)
    bg2 = ColorClip(size=(50, 50), color=(60, 60, 60)).set_duration(1).set_fps(24)
    fg = ColorClip(size=(50, 50), color=(255, 255, 255)).set_duration(1).set_fps(24)
    trans = smear_bg_crossfade_fg(
        bg1, bg2, fg, fg, 0.4, (50, 50), vec=(10, 0), fps=24, bg_brightness_dip=0.1
    )
    f_start = trans.get_frame(0)
    f_mid = trans.get_frame(0.2)
    f_end = trans.get_frame(0.4 - 1e-6)

    def lum(a):
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        return float((0.299 * r + 0.587 * g + 0.114 * b).mean())

    assert lum(f_mid) < lum(f_start)
    assert lum(f_mid) < lum(f_end)


def test_overlay_clip_no_broadcast(tmp_path):
    from PIL import Image
    import numpy as np
    cv2 = pytest.importorskip("cv2")
    from ken_burns_reel.panels import export_panels
    from ken_burns_reel.builder import make_panels_overlay_sequence

    arr = np.full((300, 200, 3), 255, np.uint8)
    cv2.rectangle(arr, (0, 10), (90, 290), (0, 0, 0), -1)
    cv2.rectangle(arr, (110, 10), (199, 290), (0, 0, 0), -1)
    page = tmp_path / "page.png"
    Image.fromarray(arr).save(page)

    mask_dir = tmp_path / "masks" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)

    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "masks"),
        target_size=(200, 300),
        dwell=0.1,
        travel=0.1,
        overlay_fit=0.95,
        overlay_margin=0,
        fg_shadow=0.25,
        fg_shadow_blur=8,
        fg_shadow_offset=3,
    )
    frame = clip.get_frame(0.05)
    assert frame.shape[:2] == (300, 200)


def test_no_broadcast_overlay_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    arr = np.full((300, 200, 3), 255, np.uint8)
    cv2.rectangle(arr, (0, 10), (90, 290), (0, 0, 0), -1)
    cv2.rectangle(arr, (110, 10), (199, 290), (0, 0, 0), -1)
    page = tmp_path / "page.png"
    Image.fromarray(arr).save(page)
    mask_dir = tmp_path / "masks" / "page_0001"
    export_panels(str(page), str(mask_dir), mode="mask", bleed=0, tight_border=0, feather=1)
    clip = make_panels_overlay_sequence(
        [str(page)],
        str(tmp_path / "masks"),
        target_size=(200, 300),
        dwell=0.1,
        travel=0.1,
        overlay_mode="anchored",
        overlay_scale=1.2,
    )
    frame = clip.get_frame(0.05)
    assert frame.shape[:2] == (300, 200)
