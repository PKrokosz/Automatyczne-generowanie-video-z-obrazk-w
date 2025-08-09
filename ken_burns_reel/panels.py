from typing import List, Tuple
import os
import cv2
import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]


def _build_panels_mask(mask_white: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gutters = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=1)
    panels_mask = cv2.bitwise_not(gutters)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    panels_mask = cv2.morphologyEx(panels_mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    return panels_mask


def detect_panels(img: Image.Image, min_area_ratio: float = 0.03) -> List[Box]:
    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    med = np.median(L)
    thr = np.clip(med + 25, 180, 245).astype(np.uint8)
    _, mask_white = cv2.threshold(L, thr, 255, cv2.THRESH_BINARY)
    white_ratio = float((mask_white == 255).mean())
    if white_ratio > 0.7:
        min_area_ratio *= 1.2
    panels_mask = _build_panels_mask(mask_white)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(panels_mask, connectivity=8)
    if num < 2:
        _, mask_white = cv2.threshold(
            L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        panels_mask = _build_panels_mask(mask_white)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            panels_mask, connectivity=8
        )
    if num < 2:
        mask_white = cv2.adaptiveThreshold(
            L,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            -5,
        )
        panels_mask = _build_panels_mask(mask_white)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            panels_mask, connectivity=8
        )
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
    return _suppress_nested(boxes)


def _suppress_nested(boxes: List[Box], thr: float = 0.85) -> List[Box]:
    keep = [True] * len(boxes)
    for i, A in enumerate(boxes):
        if not keep[i]:
            continue
        ax, ay, aw, ah = A
        area_a = aw * ah
        for j, B in enumerate(boxes):
            if i == j or not keep[j]:
                continue
            bx, by, bw, bh = B
            if ax <= bx and ay <= by and ax + aw >= bx + bw and ay + ah >= by + bh:
                area_b = bw * bh
                ratio = area_b / max(1, area_a)
                if ratio >= thr:
                    keep[j] = False
            elif bx <= ax and by <= ay and bx + bw >= ax + aw and by + bh >= ay + ah:
                area_b = bw * bh
                ratio = area_a / max(1, area_b)
                if ratio >= thr:
                    keep[i] = False
                    break
    return [b for b, k in zip(boxes, keep) if k]

def order_panels_lr_tb(boxes: List[Box], row_tol: int = 40) -> List[Box]:
    if not boxes:
        return []
    # attempt simple graph-based ordering
    n = len(boxes)
    edges = {i: set() for i in range(n)}
    indeg = {i: 0 for i in range(n)}
    for i, A in enumerate(boxes):
        Ay = A[1] + A[3] / 2
        Ax2 = A[0] + A[2]
        for j, B in enumerate(boxes):
            if i == j:
                continue
            By = B[1]
            Bx = B[0]
            if Ay <= By and Ax2 <= Bx + row_tol:
                if j not in edges[i]:
                    edges[i].add(j)
                    indeg[j] += 1
    from collections import deque

    q = deque([i for i in range(n) if indeg[i] == 0])
    order_idx: List[int] = []
    while q:
        i = q.popleft()
        order_idx.append(i)
        for j in edges[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                q.append(j)
    if len(order_idx) == n:
        return [boxes[i] for i in order_idx]

    # fallback to row/column sort
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


def export_panels(
    image_path: str,
    out_dir: str,
    mode: str = "rect",
    bleed: int = 24,
) -> List[str]:
    """Detect panels in *image_path* and export them to *out_dir*.

    Parameters
    ----------
    image_path:
        Path to the source page image.
    out_dir:
        Destination directory. A panel_<NNNN>.png file is written for each
        detected panel.
    mode:
        ``"rect"`` saves rectangular crops. ``"mask"`` saves RGBA images where
        gutters are transparent.
    bleed:
        Extra pixels around each panel crop.
    """

    os.makedirs(out_dir, exist_ok=True)

    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    med = np.median(L)
    thr = np.clip(med + 25, 180, 245).astype(np.uint8)
    _, mask_white = cv2.threshold(L, thr, 255, cv2.THRESH_BINARY)
    panels_mask = _build_panels_mask(mask_white)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        panels_mask, connectivity=8
    )
    H, W = panels_mask.shape

    comps: List[Tuple[Box, int]] = []
    min_area = int(0.03 * W * H)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / max(1, h)
        if 0.3 <= ar <= 4.0:
            comps.append(((int(x), int(y), int(w), int(h)), i))

    if not comps:
        return []

    boxes = [c[0] for c in comps]
    boxes = _suppress_nested(boxes)
    boxes = order_panels_lr_tb(boxes)

    # map boxes back to their label index
    label_map = {box: idx for box, idx in comps}
    out_paths: List[str] = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        x0 = max(0, x - bleed)
        y0 = max(0, y - bleed)
        x1 = min(W, x + w + bleed)
        y1 = min(H, y + h + bleed)
        crop = rgb[y0:y1, x0:x1]
        if mode == "mask":
            lbl = label_map[(x, y, w, h)]
            m = (labels == lbl).astype(np.uint8) * 255
            mask_crop = m[y0:y1, x0:x1]
            rgba = np.dstack([crop, mask_crop])
            im_out = Image.fromarray(rgba, mode="RGBA")
        else:
            im_out = Image.fromarray(crop, mode="RGB")
        fname = f"panel_{i:04d}.png"
        out_path = os.path.join(out_dir, fname)
        im_out.save(out_path)
        out_paths.append(out_path)
    return out_paths

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
