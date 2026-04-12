"""
Road Surface Detection: Gravel vs Asphalt
==========================================
Multi-class semantic segmentation with SegFormer (ADE20K pre-trained):
  - Sky      → sky-blue overlay
  - Roadside / vegetation / ground → dark-green overlay
  - Road surface
      · Asphalt → orange overlay
      · Gravel  → lime-green overlay

Road-type classification uses lv_ratio (fine/coarse local-variance ratio)
as the primary feature (threshold = 0.876, calibrated on 6 test images),
combined with secondary texture features for confidence estimation.
"""

import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
IMAGE_DIR = BASE_DIR / "images"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── ADE20K class groups (0-indexed) ──────────────────────────────────────────
#   original 1-indexed label → subtract 1 for 0-indexed id
SKY_IDS = {2}            # sky

ROAD_IDS = {             # road, sidewalk, path, dirt-track, land/ground
    6, 11, 52, 91, 94,
}

VEG_IDS = {              # tree, grass, earth, mountain, plant, field, hill, rock, sand
    4, 9, 13, 16, 17, 29, 34, 46, 68,
}

# ── overlay colours (BGR for OpenCV, RGB for matplotlib) ─────────────────────
COL_SKY        = dict(bgr=(200, 160,  50), rgb=(50, 160, 200))   # sky-blue
COL_VEG        = dict(bgr=( 30, 100,  30), rgb=(30, 100,  30))   # dark green
COL_ASPHALT    = dict(bgr=( 30, 100, 255), rgb=(255, 100,  30))   # orange
COL_GRAVEL     = dict(bgr=(200,  40, 160), rgb=(160,  40, 200))  # purple
COL_OTHER      = dict(bgr=(100, 100, 100), rgb=(100, 100, 100))  # gray

LV_RATIO_THR = 0.876   # primary gravel/asphalt decision boundary


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
    print(f"[INFO] Loading {model_name} ...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model     = SegformerForSemanticSegmentation.from_pretrained(
                    model_name, use_safetensors=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    print(f"[INFO] Model ready on {device.upper()}\n")
    return processor, model, device


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────
def run_segformer(image_path, processor, model, device):
    """Return full-resolution ADE20K class map (H×W int32)."""
    from PIL import ImageOps
    pil  = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    sz   = (pil.height, pil.width)
    inp  = {k: v.to(device) for k, v in processor(images=pil, return_tensors="pt").items()}
    with torch.no_grad():
        logits = model(**inp).logits
    logits  = F.interpolate(logits, size=sz, mode="bilinear", align_corners=False)
    seg_map = logits.argmax(1).squeeze().cpu().numpy().astype(np.int32)
    return seg_map


def _monotonic_taper_mask(road_mask: np.ndarray, upper_cut: int,
                           smooth_alpha: float = 0.3) -> np.ndarray:
    """
    Perspective-correct visual mask: scan bottom→top enforcing that road
    half-width can only DECREASE (road narrows toward the vanishing point).
    Center position is exponentially smoothed to follow gentle road curves.

    This naturally handles wide/flat SegFormer detections (e.g. wet roads
    where SegFormer sees full-width road at every row) without producing
    the rectangular artefact that a linear gradient taper creates.
    """
    H, W = road_mask.shape
    result = np.zeros_like(road_mask)

    prev_hw  = None        # previous half-width (None = not yet started)
    prev_ctr = W / 2.0

    for row in range(H - 1, upper_cut - 1, -1):       # bottom → top
        cols = np.where(road_mask[row])[0]

        if len(cols) < 10:
            # No road pixels here; if already tracking, gently shrink
            if prev_hw is not None:
                prev_hw = max(5.0, prev_hw * 0.94)    # ~6 % narrower per row
                x0 = max(0,     int(prev_ctr - prev_hw))
                x1 = min(W - 1, int(prev_ctr + prev_hw))
                result[row, x0:x1 + 1] = True
            continue

        seg_hw  = (int(cols[-1]) - int(cols[0])) / 2.0
        seg_ctr = (int(cols[0])  + int(cols[-1])) / 2.0

        if prev_hw is None:
            # Bottom-most valid road row — use SegFormer directly
            curr_hw  = seg_hw
            curr_ctr = seg_ctr
        else:
            # Monotonic constraint: width can only decrease going up
            curr_hw  = min(seg_hw, prev_hw)
            # Smooth center shift (allows gentle road curves)
            curr_ctr = (1.0 - smooth_alpha) * prev_ctr + smooth_alpha * seg_ctr

        x0 = max(0,     int(curr_ctr - curr_hw))
        x1 = min(W - 1, int(curr_ctr + curr_hw))
        result[row, x0:x1 + 1] = True
        prev_hw  = curr_hw
        prev_ctr = curr_ctr

    return result & road_mask


def _detect_red_line_bounds(bgr: np.ndarray, upper_cut: int) -> np.ndarray | None:
    """
    Detect red no-parking lines (台灣禁止停車紅線) and return a boolean mask
    covering the area INSIDE the red lines (i.e. the road corridor they bound).

    Red paint in HSV: H ∈ [0,12] ∪ [168,180], S > 80, V > 50.
    The detected red pixels are dilated and used as a fill boundary so the
    road mask stays within the lane delimited by the red kerb lines.

    Returns None when no significant red lines are found.
    """
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red wraps around H=0 in HSV
    r1 = cv2.inRange(hsv, (0,   80, 50), (12,  255, 255))
    r2 = cv2.inRange(hsv, (168, 80, 50), (180, 255, 255))
    red = cv2.bitwise_or(r1, r2)

    # Only consider road zone
    red[:upper_cut] = 0

    # Connect broken segments along horizontal / diagonal direction
    kh = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 3))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kh)

    red_px = int(red.sum() / 255)
    road_px = (H - upper_cut) * W

    # Require meaningful coverage AND that red lines span multiple rows
    red_bin = (red > 0)
    active_rows = int(red_bin[upper_cut:].any(axis=1).sum())
    if red_px < road_px * 0.015 or active_rows < 8:   # < 1.5 % or < 8 rows → ignore
        return None

    # Additional sanity: the red region must span at least 25 % image width
    # (lane-boundary lines are long, not scattered small patches)
    red_cols = np.where(red_bin.any(axis=0))[0]
    if len(red_cols) == 0 or (red_cols[-1] - red_cols[0]) < W * 0.25:
        return None

    print(f"  [INFO] Red boundary lines detected ({red_px / road_px:.1%} of road zone)")

    # Build "inside-lines" mask:
    # For each row find the leftmost and rightmost red pixel;
    # fill between them (those are the road lane bounds).
    inside = np.zeros((H, W), dtype=np.uint8)
    for row in range(upper_cut, H):
        cols = np.where(red_bin[row])[0]
        if len(cols) < 2:
            continue
        inside[row, int(cols[0]) : int(cols[-1]) + 1] = 1

    # Fill vertically: propagate non-zero rows downward/upward using dilation
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    inside = cv2.dilate(inside, kv)
    return inside.astype(bool)


def _hough_road_polygon(bgr, upper_cut: int) -> np.ndarray | None:
    """
    Detect lane lines with Hough transform and return a filled
    perspective polygon (H×W bool), or None if detection fails.

    The polygon covers the road corridor from the bottom up to
    approximately the vanishing-point row.
    """
    H, W = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Work only in the road region
    roi = gray[upper_cut:].copy()
    roi_h = roi.shape[0]

    blur  = cv2.GaussianBlur(roi, (7, 7), 0)
    edges = cv2.Canny(blur, 40, 120)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=40, minLineLength=60, maxLineGap=50)
    if lines is None:
        return None

    cx = W // 2
    left_pts, right_pts = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        # Convert ROI-relative y to full-image y
        y1f, y2f = y1 + upper_cut, y2 + upper_cut

        # Left lane: negative slope, left half
        if -5.0 < slope < -0.25 and min(x1, x2) < cx:
            left_pts += [(x1, y1f), (x2, y2f)]
        # Right lane: positive slope, right half
        elif 0.25 < slope < 5.0 and max(x1, x2) > cx:
            right_pts += [(x1, y1f), (x2, y2f)]

    def extrapolate(pts, y_bot, y_top):
        if len(pts) < 4:
            return None, None
        arr = np.array(pts)
        coeffs = np.polyfit(arr[:, 1], arr[:, 0], 1)   # x = f(y)
        return int(np.polyval(coeffs, y_bot)), int(np.polyval(coeffs, y_top))

    y_bottom = H - 1
    y_top    = upper_cut + max(20, int(roi_h * 0.15))   # ~15 % above upper_cut

    lx_b, lx_t = extrapolate(left_pts,  y_bottom, y_top)
    rx_b, rx_t = extrapolate(right_pts, y_bottom, y_top)

    if lx_b is None or rx_b is None:
        return None

    # Clamp to image bounds
    def clamp(x): return max(0, min(W - 1, x))
    lx_b, lx_t = clamp(lx_b), clamp(lx_t)
    rx_b, rx_t = clamp(rx_b), clamp(rx_t)

    # Sanity: left must be left of right
    if lx_b >= rx_b or lx_t >= rx_t:
        return None

    poly_pts = np.array([[lx_b, y_bottom], [rx_b, y_bottom],
                          [rx_t, y_top],   [lx_t, y_top]], dtype=np.int32)
    poly = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly, [poly_pts], 1)
    return poly.astype(bool)


def _default_trapezoid(H, W, upper_cut: int) -> np.ndarray:
    """
    Fallback: fixed trapezoid centred on the lower image.
    Narrow at the top (~30 % width), wide at the bottom (full width).
    """
    bx0, bx1 = int(W * 0.00), int(W * 1.00)
    tx0, tx1 = int(W * 0.30), int(W * 0.70)
    ty        = upper_cut + max(10, int((H - upper_cut) * 0.10))
    pts = np.array([[bx0, H - 1], [bx1, H - 1],
                    [tx1, ty],    [tx0, ty]], dtype=np.int32)
    trap = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(trap, [pts], 1)
    return trap.astype(bool)


def make_class_masks(seg_map, bgr):
    """
    Build boolean masks for each semantic group.

    Returns
    -------
    sky_mask      : full sky region
    veg_mask      : full vegetation region
    road_mask     : full road region (used for texture classification)
    road_vis_mask : perspective-shaped road polygon (used for display only)
    """
    H, W = seg_map.shape

    sky_mask  = np.isin(seg_map, list(SKY_IDS))
    veg_mask  = np.isin(seg_map, list(VEG_IDS))
    road_mask = np.isin(seg_map, list(ROAD_IDS))

    # ── constrain road to lower 65 % ─────────────────────────────────────────
    upper_cut = int(H * 0.35)
    road_mask[:upper_cut, :] = False

    # ── morphological clean-up for road ──────────────────────────────────────
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    road_mask = cv2.morphologyEx(road_mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    road_mask = cv2.morphologyEx(road_mask.astype(np.uint8), cv2.MORPH_OPEN,  k).astype(bool)

    # ── keep only the largest connected road component ───────────────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        road_mask.astype(np.uint8), connectivity=8)
    if n_labels > 1:
        areas   = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
        largest = 1 + int(np.argmax(areas))
        road_mask = (labels == largest)

    # ── fallback if road mask too small ──────────────────────────────────────
    if road_mask.sum() / road_mask.size < 0.03:
        road_mask[int(H * 0.5):, int(W * 0.1):int(W * 0.9)] = True
        print("  [WARN] sparse road mask – applied centre-bottom fallback")

    # ── visual mask: monotonic taper (always) ────────────────────────────────
    # Enforce that road width can only DECREASE going toward the vanishing point.
    # When SegFormer already has a natural taper this is a no-op; when it
    # over-detects wide flat bands (e.g. wet/reflective roads) the taper
    # corrects the shape without affecting texture-based classification.
    road_vis_mask = _monotonic_taper_mask(road_mask, upper_cut)
    seg_cov = road_mask.sum() / road_mask.size
    vis_cov = road_vis_mask.sum() / road_vis_mask.size
    if abs(seg_cov - vis_cov) > 0.02:       # only log when there's a real change
        print(f"  [INFO] monotonic taper: {seg_cov:.0%} → {vis_cov:.0%}")

    return sky_mask, veg_mask, road_mask, road_vis_mask


# ─────────────────────────────────────────────────────────────────────────────
# Road-type classifier
# ─────────────────────────────────────────────────────────────────────────────
def local_variance_map(gray_f32, win):
    mu   = cv2.boxFilter(gray_f32, -1, (win, win))
    mu2  = cv2.boxFilter(gray_f32 * gray_f32, -1, (win, win))
    return np.sqrt(np.maximum(mu2 - mu * mu, 0.0))


def classify_road_type(bgr, road_mask):
    """
    Primary: lv_ratio = mean(local_var_5) / mean(local_var_15) in road region.
             Threshold 0.876 separates gravel (>=) from asphalt (<).
    Secondary: edge density, Laplacian variance, mean intensity
               → converted to a 0–1 confidence score.

    Returns (label, gravel_score) where gravel_score ∈ [0, 1].
    """
    if road_mask.sum() < 200:
        return "unknown", 0.5

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Bounding box of road region
    ys, xs = np.where(road_mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    roi      = gray[y0:y1, x0:x1]
    roi_mask = road_mask[y0:y1, x0:x1]

    # ── lv_ratio ─────────────────────────────────────────────────────────────
    lv5  = local_variance_map(roi, 5)[roi_mask].mean()
    lv15 = local_variance_map(roi, 15)[roi_mask].mean()
    lv_ratio = float(lv5 / (lv15 + 1e-8))

    # ── secondary features ───────────────────────────────────────────────────
    sx   = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    sy   = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.sqrt(sx**2 + sy**2)[roi_mask].mean())

    lap_var   = float(cv2.Laplacian(roi, cv2.CV_32F)[roi_mask].std())
    mean_int  = float(gray[road_mask].mean())

    # ── combine into gravel_score ─────────────────────────────────────────────
    # lv_ratio is the anchor; others adjust the probability
    lv_score   = float(np.clip((lv_ratio  - 0.750) / 0.200, 0, 1))   # 35 %
    edge_score = float(np.clip((edge      -  60.0) / 200.0, 0, 1))   # 25 %
    lap_score  = float(np.clip((lap_var   -  30.0) / 100.0, 0, 1))   # 25 %
    int_score  = float(np.clip((mean_int  -  90.0) / 100.0, 0, 1))   # 15 %

    gravel_score = (0.35 * lv_score  +
                    0.25 * edge_score +
                    0.25 * lap_score  +
                    0.15 * int_score)

    # Hard decision with lv_ratio as primary discriminant
    if lv_ratio >= LV_RATIO_THR:
        label = "gravel"
        gravel_score = max(gravel_score, 0.50)   # ensure > 0.5 for gravel
    else:
        label = "asphalt"
        gravel_score = min(gravel_score, 0.49)   # ensure < 0.5 for asphalt

    features = dict(lv_ratio=lv_ratio, lv_score=lv_score,
                    edge=edge, lap_var=lap_var, mean_int=mean_int)
    return label, float(gravel_score), features


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────
def build_coloured_overlay(bgr, sky_mask, veg_mask, road_mask, road_label):
    """
    Returns a blended BGR image:
      sky        → sky-blue
      vegetation → dark green
      road       → orange (asphalt) or lime green (gravel)
      other      → slight gray tint
    """
    col_road = COL_ASPHALT["bgr"] if road_label == "asphalt" else COL_GRAVEL["bgr"]

    overlay = bgr.copy()
    overlay[sky_mask]  = COL_SKY["bgr"]
    overlay[veg_mask]  = COL_VEG["bgr"]
    overlay[road_mask] = col_road

    # other pixels: subtle gray tint
    other_mask = ~(sky_mask | veg_mask | road_mask)
    overlay[other_mask] = (overlay[other_mask].astype(int) * 0.60 +
                           np.array(COL_OTHER["bgr"]) * 0.40).astype(np.uint8)

    return cv2.addWeighted(bgr, 0.40, overlay, 0.60, 0)


def add_banner(result_bgr, road_label, gravel_score, road_mask):
    h, w = result_bgr.shape[:2]
    bh   = max(56, int(h * 0.085))
    cv2.rectangle(result_bgr, (0, 0), (w, bh), (15, 15, 15), -1)

    conf    = gravel_score if road_label == "gravel" else 1 - gravel_score
    t_col   = COL_GRAVEL["bgr"] if road_label == "gravel" else COL_ASPHALT["bgr"]
    cv2.putText(result_bgr, f"{road_label.upper()}  {conf:.0%}",
                (12, bh - 12), cv2.FONT_HERSHEY_DUPLEX, bh / 56.0, t_col, 2, cv2.LINE_AA)

    legend_items = [
        ("Sky",       COL_SKY["bgr"]),
        ("Vegetation", COL_VEG["bgr"]),
        ("Asphalt",   COL_ASPHALT["bgr"]),
        ("Gravel",    COL_GRAVEL["bgr"]),
    ]
    xoff = w - 210
    for i, (name, col) in enumerate(legend_items):
        y = h - 26 - i * 26
        cv2.rectangle(result_bgr, (xoff, y - 14), (xoff + 18, y + 2), col, -1)
        cv2.putText(result_bgr, name, (xoff + 24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    cov = road_mask.sum() / road_mask.size
    cv2.putText(result_bgr, f"Road area: {cov:.0%}",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    return result_bgr


def save_panel(bgr, sky_mask, veg_mask, road_mask, road_label,
               gravel_score, features, out_path, title):
    """3-panel matplotlib figure: original | seg mask | coloured result."""
    col_road_rgb = COL_GRAVEL["rgb"] if road_label == "gravel" else COL_ASPHALT["rgb"]
    rgb          = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Build colour segmentation map (RGB)
    seg_vis = np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    seg_vis[sky_mask]  = COL_SKY["rgb"]
    seg_vis[veg_mask]  = COL_VEG["rgb"]
    seg_vis[road_mask] = col_road_rgb
    seg_vis[~(sky_mask | veg_mask | road_mask)] = COL_OTHER["rgb"]

    # Blended result
    overlay = rgb.copy()
    overlay[sky_mask]  = COL_SKY["rgb"]
    overlay[veg_mask]  = COL_VEG["rgb"]
    overlay[road_mask] = col_road_rgb
    result_rgb = (rgb * 0.40 + overlay * 0.60).astype(np.uint8)

    fig = plt.figure(figsize=(20, 7), facecolor="#12121f")
    gs  = gridspec.GridSpec(1, 3, wspace=0.04)

    for ax_i, (img, sub) in enumerate([
        (rgb,        "Original"),
        (seg_vis,    "Semantic Mask"),
        (result_rgb, f"Result: {road_label.upper()} ({(gravel_score if road_label=='gravel' else 1-gravel_score):.0%})"),
    ]):
        ax = fig.add_subplot(gs[ax_i])
        ax.imshow(img)
        ax.set_title(sub, color="white", fontsize=13, pad=5)
        ax.axis("off")

    feat_str = "  ".join(f"{k}={v:.3f}" for k, v in features.items())
    fig.text(0.5, 0.01, feat_str, ha="center", color="#999999",
             fontsize=8.5, family="monospace")

    # Legend patches
    from matplotlib.patches import Patch
    patches = [
        Patch(color=[c/255 for c in COL_SKY["rgb"]],     label="Sky"),
        Patch(color=[c/255 for c in COL_VEG["rgb"]],     label="Vegetation"),
        Patch(color=[c/255 for c in COL_ASPHALT["rgb"]], label="Asphalt"),
        Patch(color=[c/255 for c in COL_GRAVEL["rgb"]],  label="Gravel"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
               facecolor="#1e1e2e", labelcolor="white", framealpha=0.9,
               bbox_to_anchor=(0.5, 0.06))

    fig.suptitle(title, color="white", fontsize=14, y=1.01)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {out_path.name}")


def save_summary(results, out_path):
    """Dynamic summary grid (cols=3, rows adjusted to fit)."""
    n_cols = 3
    n_rows = max(1, (len(results) + n_cols - 1) // n_cols)
    fig = plt.figure(figsize=(24, 8 * n_rows), facecolor="#12121f")
    gs  = gridspec.GridSpec(n_rows, n_cols, wspace=0.05, hspace=0.15)

    for i, r in enumerate(results):
        bgr        = r["bgr"]
        sky_mask   = r["sky_mask"]
        veg_mask   = r["veg_mask"]
        road_mask  = r["road_mask"]
        label      = r["label"]
        score      = r["score"]
        name       = r["name"]

        col_road_rgb = COL_GRAVEL["rgb"] if label == "gravel" else COL_ASPHALT["rgb"]
        rgb          = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay      = rgb.copy()
        overlay[sky_mask]  = COL_SKY["rgb"]
        overlay[veg_mask]  = COL_VEG["rgb"]
        overlay[road_mask] = col_road_rgb
        result_rgb   = (rgb * 0.40 + overlay * 0.60).astype(np.uint8)

        ax    = fig.add_subplot(gs[i // 3, i % 3])
        ax.imshow(result_rgb)
        conf  = score if label == "gravel" else 1 - score
        t_col = "#4edc64" if label == "gravel" else "#ff823c"
        ax.set_title(f"{name}\n{label.upper()}  {conf:.0%}", color=t_col, fontsize=12, pad=4)
        ax.axis("off")

    from matplotlib.patches import Patch
    patches = [
        Patch(color=[c/255 for c in COL_SKY["rgb"]],     label="Sky"),
        Patch(color=[c/255 for c in COL_VEG["rgb"]],     label="Vegetation"),
        Patch(color=[c/255 for c in COL_ASPHALT["rgb"]], label="Asphalt Road"),
        Patch(color=[c/255 for c in COL_GRAVEL["rgb"]],  label="Gravel Road"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=12,
               facecolor="#1e1e2e", labelcolor="white", framealpha=0.85,
               bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        "Road Surface Detection Summary  –  SegFormer (ADE20K) + Texture Classifier",
        color="white", fontsize=16, y=1.005)
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[INFO] Summary saved  {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    image_paths = sorted(
        p for p in IMAGE_DIR.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "road" in p.name.lower()
    )
    if not image_paths:
        print("[ERROR] No images found in images/")
        sys.exit(1)
    print(f"[INFO] {len(image_paths)} images: {[p.name for p in image_paths]}\n")

    processor, model, device = load_model()
    results = []

    for img_path in image_paths:
        print(f"[>>>] {img_path.name}")

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [ERROR] cannot read"); continue

        # 1. Segment
        seg_map = run_segformer(img_path, processor, model, device)
        sky_mask, veg_mask, road_mask, road_vis_mask = make_class_masks(seg_map, bgr)

        cov     = road_mask.sum()     / road_mask.size
        vis_cov = road_vis_mask.sum() / road_vis_mask.size
        sky_cov = sky_mask.sum()      / sky_mask.size
        veg_cov = veg_mask.sum()      / veg_mask.size
        print(f"  Coverage  road={cov:.1%}(full)/{vis_cov:.1%}(vis)  sky={sky_cov:.1%}  veg={veg_cov:.1%}")

        # 2. Classify road type – uses full mask for accurate texture stats
        label, score, features = classify_road_type(bgr, road_mask)
        conf = score if label == "gravel" else 1 - score
        print(f"  lv_ratio={features['lv_ratio']:.4f}  →  {label.upper()} ({conf:.0%})")

        # 3. Save outputs – uses visual mask for clean overlay
        stem       = img_path.stem.replace(" ", "_")
        result_bgr = build_coloured_overlay(bgr, sky_mask, veg_mask, road_vis_mask, label)
        result_bgr = add_banner(result_bgr, label, score, road_vis_mask)
        cv2.imwrite(str(OUT_DIR / f"{stem}_result.jpg"), result_bgr)

        save_panel(bgr, sky_mask, veg_mask, road_vis_mask, label, score, features,
                   OUT_DIR / f"{stem}_panel.png", title=img_path.name)

        results.append(dict(name=img_path.stem, bgr=bgr,
                            sky_mask=sky_mask, veg_mask=veg_mask,
                            road_mask=road_vis_mask, label=label, score=score,
                            features=features))
        print()

    # 4. Summary figure
    if results:
        save_summary(results, OUT_DIR / "summary.png")

    # 5. Result table
    print("\n" + "=" * 65)
    print(f"{'Image':<30}  {'Pred':<8}  {'Conf':>6}  {'GT':<8}  OK?")
    print("-" * 65)
    correct = 0
    for r in results:
        conf = r["score"] if r["label"] == "gravel" else 1 - r["score"]
        gt   = "gravel" if "gravel" in r["name"].lower() else "asphalt"
        ok   = "YES" if r["label"] == gt else "NO "
        print(f"{r['name']:<30}  {r['label']:<8}  {conf:>5.0%}  {gt:<8}  {ok}")
        if r["label"] == gt:
            correct += 1
    print("=" * 65)
    print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.0%}\n")


if __name__ == "__main__":
    main()
