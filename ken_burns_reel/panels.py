from typing import List, Tuple
import os
import cv2
import numpy as np
from PIL import Image
from .utils import gaussian_blur

Box = Tuple[int, int, int, int]


def _build_panels_mask(mask_white: np.ndarray, gutter_thicken: int = 0) -> np.ndarray:
    if gutter_thicken > 0:
        kernel_g = cv2.getStructuringElement(
            cv2.MORPH_RECT, (gutter_thicken, gutter_thicken)
        )
        mask_white = cv2.dilate(mask_white, kernel_g, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gutters = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=1)
    panels_mask = cv2.bitwise_not(gutters)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    panels_mask = cv2.morphologyEx(panels_mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    return panels_mask


def alpha_bbox(arr: np.ndarray) -> Box:
    """Return bounding box of non-zero alpha in RGBA array."""
    if arr.shape[-1] < 4:
        h, w = arr.shape[:2]
        return (0, 0, w, h)
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        h, w = alpha.shape
        return (0, 0, w, h)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside a binary mask using flood fill from the border."""
    h, w = mask.shape[:2]
    flood = np.zeros((h + 2, w + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood, (0, 0), 255)
    inv = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, inv)


def roughen_alpha(mask: np.ndarray, amount: float, scale: int) -> np.ndarray:
    """Add small irregularities to the mask edge."""
    if amount <= 0:
        return mask
    h, w = mask.shape[:2]
    edge = cv2.Canny(mask, 50, 150)
    edge = cv2.dilate(edge, None, iterations=1)
    noise = (np.random.rand(h, w).astype(np.float32) - 0.5)
    noise = cv2.GaussianBlur(noise, (0, 0), max(1, scale))
    perturb = noise * (amount * 255.0) * (edge / 255.0)
    out = mask.astype(np.float32) + perturb
    out = cv2.GaussianBlur(out, (0, 0), 0.66 + amount * 1.5)
    return np.clip(out, 0, 255).astype(np.uint8)


def detect_panels(
    img: Image.Image,
    min_area_ratio: float = 0.03,
    gutter_thicken: int = 0,
    min_ar: float = 0.4,
    max_ar: float = 2.8,
) -> List[Box]:
    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    med = np.median(L)
    thr = np.clip(med + 25, 180, 245).astype(np.uint8)
    _, mask_white = cv2.threshold(L, thr, 255, cv2.THRESH_BINARY)
    white_ratio = float((mask_white == 255).mean())
    if white_ratio > 0.7:
        min_area_ratio *= 1.2
    panels_mask = _build_panels_mask(mask_white, gutter_thicken)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        panels_mask, connectivity=8
    )
    if num < 2 and white_ratio > 0.7:
        thr2 = np.clip(med + 18, 180, 245).astype(np.uint8)
        _, mask_white = cv2.threshold(L, thr2, 255, cv2.THRESH_BINARY)
        panels_mask = _build_panels_mask(mask_white, gutter_thicken)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            panels_mask, connectivity=8
        )
    if num < 2:
        _, mask_white = cv2.threshold(
            L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        panels_mask = _build_panels_mask(mask_white, gutter_thicken)
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
        panels_mask = _build_panels_mask(mask_white, gutter_thicken)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            panels_mask, connectivity=8
        )
    H, W = panels_mask.shape
    min_area = int(min_area_ratio * W * H)
    boxes: List[Box] = []
    total_comps = num - 1
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / max(1, h)
        if total_comps > 1 and not (min_ar <= ar <= max_ar):
            continue
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
    tight_border: int = 1,
    feather: int = 1,
    roughen: float = 0.15,
    roughen_scale: int = 24,
    gutter_thicken: int = 0,
    min_area_ratio: float = 0.03,
    mask_fill_holes: int = 1,
    mask_close: int = 5,
    mask_rect_fallback: float = 0.12,
    panel_min_ar: float = 0.4,
    panel_max_ar: float = 2.8,
) -> List[str]:
    """Detect panels in *image_path* and export them to *out_dir*.

    Parameters
    ----------
    image_path:
        Path to the source page image.
    out_dir:
        Destination directory. A ``panel_<NNNN>.png`` file is written for each
        detected panel.
    mode:
        ``"rect"`` saves rectangular crops. ``"mask"`` saves RGBA images where
        gutters are transparent.
    bleed:
        Extra pixels around each panel crop.
    tight_border:
        Number of pixels to erode from the panel mask edge when ``mode="mask"``.
    feather:
        Radius for Gaussian blur applied to the alpha mask when ``mode="mask"``.
    roughen / roughen_scale:
        Amount and scale of edge noise applied to the mask when ``mode="mask"``.
    mask_fill_holes:
        When non-zero, fill holes in the panel mask using flood fill.
    mask_close:
        Kernel size for ``cv2.MORPH_CLOSE`` applied to the panel mask.
    mask_rect_fallback:
        If the ratio of holes in the mask after feathering exceeds this value,
        fall back to a rectangular crop with solid alpha.
    panel_min_ar / panel_max_ar:
        Aspect-ratio bounds for detected components (ignored if only one component).
    """

    os.makedirs(out_dir, exist_ok=True)

    with Image.open(image_path) as im:
        rgb = np.array(im.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]
    med = np.median(L)
    thr = np.clip(med + 25, 180, 245).astype(np.uint8)
    _, mask_white = cv2.threshold(L, thr, 255, cv2.THRESH_BINARY)
    white_ratio = float((mask_white == 255).mean())
    if white_ratio > 0.7:
        min_area_ratio *= 1.2
    panels_mask = _build_panels_mask(mask_white, gutter_thicken)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        panels_mask, connectivity=8
    )
    H, W = panels_mask.shape

    comps: List[Tuple[Box, int]] = []
    min_area = int(min_area_ratio * W * H)
    total_comps = num - 1
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ar = w / max(1, h)
        if total_comps > 1 and not (panel_min_ar <= ar <= panel_max_ar):
            continue
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
            if mask_fill_holes:
                mask_crop = fill_holes(mask_crop)
            if mask_close > 0:
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_close, mask_close))
                mask_crop = cv2.morphologyEx(mask_crop, cv2.MORPH_CLOSE, k, iterations=1)
            if tight_border > 0:
                kernel = np.ones((tight_border, tight_border), np.uint8)
                mask_crop = cv2.erode(mask_crop, kernel, iterations=1)
            if feather > 0:
                mask_crop = gaussian_blur(mask_crop, sigma=feather)
            if roughen > 0:
                mask_crop = roughen_alpha(mask_crop, roughen, roughen_scale)
            alpha = mask_crop
            bin_alpha = (alpha > 127).astype(np.uint8)
            holes_ratio = 1.0 - float(bin_alpha.sum()) / bin_alpha.size
            if holes_ratio > mask_rect_fallback:
                alpha = np.full_like(alpha, 255)
            rgba = np.dstack([crop, alpha])
            im_out = Image.fromarray(rgba, mode="RGBA")
        else:
            alpha = np.full((crop.shape[0], crop.shape[1], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([crop, alpha], axis=2)
            im_out = Image.fromarray(rgba, mode="RGBA")
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
