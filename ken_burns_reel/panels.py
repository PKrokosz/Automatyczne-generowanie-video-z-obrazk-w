from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]

def detect_panels(img: Image.Image, min_area_ratio: float = 0.03, white_thr: int = 235) -> List[Box]:
    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    _, mask_white = cv2.threshold(L, white_thr, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gutters = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=1)
    panels_mask = cv2.bitwise_not(gutters)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    panels_mask = cv2.morphologyEx(panels_mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(panels_mask, connectivity=8)
    H, W = panels_mask.shape
    min_area = int(min_area_ratio * W * H)
    boxes: List[Box] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / max(1, h)
        if 0.3 <= ar <= 4.0:
            boxes.append((int(x), int(y), int(w), int(h)))
    return boxes

def order_panels_lr_tb(boxes: List[Box], row_tol: int = 40) -> List[Box]:
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    rows: List[List[Box]] = []
    for b in boxes:
        placed = False
        for row in rows:
            if abs(row[0][1] - b[1]) <= row_tol:
                row.append(b)
                placed = True
                break
        if not placed:
            rows.append([b])
    out: List[Box] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b[0]))
    return out

def debug_detect_panels(folder: str) -> None:
    import os

    images = [
        f
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise FileNotFoundError("Brak obrazów do debugowania.")
    image_path = os.path.join(folder, images[0])
    with Image.open(image_path) as img:
        boxes = order_panels_lr_tb(detect_panels(img))
        vis = np.array(img.convert("RGB")).copy()
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            vis,
            str(i + 1),
            (x + 10, y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
    out_path = os.path.join(folder, "panels_debug.jpg")
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
