from PIL import Image

from ken_burns_reel.focus import detect_focus_point


def test_black_frame_returns_center():
    img = Image.new("RGB", (100, 100), "black")
    assert detect_focus_point(img) == (50, 50)
