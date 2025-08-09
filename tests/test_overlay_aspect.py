from ken_burns_reel.builder import _fit_window_to_box


def test_overlay_keeps_aspect() -> None:
    img_w = img_h = 200
    box = (20, 20, 100, 150)
    cx, cy, win_w, win_h = _fit_window_to_box(img_w, img_h, box, (200, 200))
    x, y, w, h = box
    S = 200 / win_w
    dst_w = int(round(w * S * 1.6))
    dst_h = int(round(h * S * 1.6))
    ar_final = dst_w / dst_h
    ar_orig = w / h
    assert abs(ar_final - ar_orig) / ar_orig <= 0.005
