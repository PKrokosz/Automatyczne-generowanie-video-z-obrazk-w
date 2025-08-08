import numpy as np
from PIL import Image

from ken_burns_reel.focus import detect_focus_point


def test_detect_focus_point_center():
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[40:60, 40:60] = 255
    img = Image.fromarray(arr)
    x, y = detect_focus_point(img)
    assert 40 <= x <= 60
    assert 40 <= y <= 60
