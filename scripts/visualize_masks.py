import json
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog


# ===== GUI Pickers =====
def pick_image_file() -> Path:
    """Pick any image file to determine the images folder."""
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select any image from the dataset",
        filetypes=(
            ("Image files", "*.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ),
    )
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)


def pick_json_file() -> Path:
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select the segmented JSON",
        filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
    )
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)


def get_screen_size():
    root = Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


# ===== COCO Helpers =====
def read_coco_grouped(json_path: Path):
    """Load COCO JSON and group annotations by image_id."""
    with open(json_path, "r") as f:
        coco = json.load(f)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    images_sorted = sorted(images, key=lambda im: im["file_name"])
    anns_per_img = {}
    for ann in annotations:
        anns_per_img.setdefault(ann["image_id"], []).append(ann)
    return images_sorted, anns_per_img


def stable_id(ann):
    """Generate stable ID for annotation (uses track_id if available)."""
    tid = ann.get("attributes", {}).get("track_id", None)
    return f"track_{tid}" if tid is not None else f"ann_{ann['id']}"


def color_for_sid(sid):
    """Generate consistent color for a given stable ID."""
    h = abs(hash(sid)) % 360
    hsv = np.uint8([[[h // 2, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ===== RLE Decoding =====
def rle_decode(seg):
    """Decode COCO-style RLE segmentation to binary mask."""
    try:
        from pycocotools import mask as maskUtils
    except ImportError:
        maskUtils = None

    # Multipart: list of RLE dicts
    if isinstance(seg, list):
        if maskUtils is None:
            raise RuntimeError(
                "Multipart compressed RLE requires pycocotools. "
                "Install with: pip install pycocotools"
            )
        m = maskUtils.decode(seg)  # (H,W,N)
        if m.ndim == 3:
            m = (m.max(axis=2) > 0).astype(np.uint8)  # union of parts
        elif m.ndim == 2:
            m = (m > 0).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected decoded shape for list RLE: {m.shape}")
        return np.ascontiguousarray(m.astype(np.uint8))

    # Single RLE dict
    if not isinstance(seg, dict):
        raise TypeError(f"Unsupported segmentation type: {type(seg)}")

    H, W = seg["size"]
    counts = seg["counts"]

    # Compressed RLE (string)
    if isinstance(counts, str):
        if maskUtils is None:
            raise RuntimeError(
                "Compressed RLE requires pycocotools. "
                "Install with: pip install pycocotools"
            )
        rle = {"size": [int(H), int(W)], "counts": counts.encode("ascii")}
        m = maskUtils.decode(rle)  # (H,W) or (H,W,1)
        if m.ndim == 3:
            m = m[:, :, 0]
        elif m.ndim != 2:
            raise ValueError(f"Unexpected decoded shape for compressed RLE: {m.shape}")
        return np.ascontiguousarray((m > 0).astype(np.uint8))

    # Uncompressed RLE (counts list of runs)
    if isinstance(counts, list):
        runs = counts
        total = int(H) * int(W)
        flat = np.zeros(total, dtype=np.uint8)
        val = 0
        idx = 0
        for run_len in runs:
            rl = int(run_len)
            if rl > 0 and val == 1:
                flat[idx:idx + rl] = 1
            idx += rl
            val ^= 1
        if idx < total:
            flat = np.pad(flat, (0, total - idx), constant_values=0)
        m = flat.reshape((int(H), int(W)), order="F")
        return np.ascontiguousarray(m.astype(np.uint8))

    raise TypeError(f"Unsupported RLE 'counts' type: {type(counts)}")


# ===== Visualization =====
def overlay_masks(frame_bgr, masks, colors, alpha=0.45):
    """Overlay colored masks on the image."""
    out = frame_bgr.copy()
    H, W = out.shape[:2]
    for m, color in zip(masks, colors):
        if m is None:
            continue
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m = (m > 0).astype(np.uint8)
        if m.sum() == 0:
            continue
        color_img = np.zeros((H, W, 3), dtype=np.uint8)
        color_img[:] = color
        out = np.where(
            m[..., None] == 1,
            (alpha * color_img + (1 - alpha) * out).astype(np.uint8),
            out
        )
    return out


def draw_boxes(frame_bgr, anns, colors=None):
    """Draw bounding boxes with labels."""
    vis = frame_bgr.copy()
    for i, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        color = (0, 255, 0) if colors is None else colors[i]
        lbl = stable_id(ann).replace("track_", "t=").replace("ann_", "a=")
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text = max(0, y0 - 4)
        cv2.rectangle(vis, (x0, y_text - th - 4), (x0 + tw + 6, y_text), color, -1)
        cv2.putText(
            vis, lbl, (x0 + 3, y_text - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return vis


# ===== Main Playback =====
def main():
    # Pick any image to determine the folder
    sample_image = pick_image_file()
    images_folder = sample_image.parent
    
    # Try to find segmented JSON automatically
    candidate = images_folder.parent / "labels_with_segmentation" / "segmented.json"
    if candidate.exists():
        json_path = candidate
        print(f"Auto-detected JSON: {json_path}")
    else:
        json_path = pick_json_file()

    print(f"Images folder: {images_folder}")
    print(f"JSON: {json_path}")

    images_sorted, anns_per_img = read_coco_grouped(json_path)

    # Setup window
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * 0.9)
    max_h = int(screen_h * 0.9)
    win = "Mask Visualization (q=quit, n=next, p=prev)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    mask_cache = {}
    idx = 0
    total = len(images_sorted)

    print(f"\nLoaded {total} images. Use 'n'=next, 'p'=prev, 'q'=quit")

    while True:
        img_info = images_sorted[idx]
        img_path = images_folder / img_info["file_name"]

        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read image: {img_path}")
            idx = (idx + 1) % total
            continue

        H, W = frame.shape[:2]

        # Resize window to fit screen
        scale = min(max_w / max(W, 1), max_h / max(H, 1), 2.0)
        disp_w, disp_h = int(W * scale), int(H * scale)
        cv2.resizeWindow(win, disp_w, disp_h)

        # Get annotations for this image
        img_id = img_info["id"]
        anns = anns_per_img.get(img_id, [])

        # Generate colors
        colors = [color_for_sid(stable_id(a)) for a in anns]
        masks = []

        # Decode masks
        for a in anns:
            rle = a.get("segmentation")
            if not rle:
                masks.append(None)
                continue

            ann_id = a["id"]
            if ann_id not in mask_cache:
                try:
                    mask_cache[ann_id] = rle_decode(rle)
                except Exception as e:
                    print(f"Decode failed for annotation {ann_id}: {e}")
                    mask_cache[ann_id] = None

            masks.append(mask_cache[ann_id])

        # Visualize
        vis = overlay_masks(frame, masks, colors, alpha=0.45)
        vis = draw_boxes(vis, anns, colors=colors)
        
        # Add image info text
        info_text = f"Image {idx+1}/{total}: {img_info['file_name']} | {len(anns)} objects"
        cv2.putText(
            vis, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            vis, info_text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA
        )
        
        cv2.imshow(win, vis)

        # Handle keyboard input
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('q'), 27):  # q or ESC
            break
        elif k == ord('n'):  # next
            idx = (idx + 1) % total
        elif k == ord('p'):  # previous
            idx = (idx - 1 + total) % total

    cv2.destroyAllWindows()
    print("Visualization closed.")


if __name__ == "__main__":
    main()