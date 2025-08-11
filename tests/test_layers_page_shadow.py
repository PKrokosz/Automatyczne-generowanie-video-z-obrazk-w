import numpy as np
from ken_burns_reel.layers import page_shadow

def test_page_shadow_generates_shadow() -> None:
    img = np.zeros((20, 20, 4), dtype=np.uint8)
    img[5:15, 5:15, :3] = 255
    img[5:15, 5:15, 3] = 255
    out = page_shadow(img, strength=0.5, blur=3, offset_xy=(2, 2))
    assert out.shape[0] > img.shape[0]
    assert out.shape[1] > img.shape[1]
    alpha = out[:, :, 3]
    assert ((alpha > 0) & (alpha < 255)).any()
