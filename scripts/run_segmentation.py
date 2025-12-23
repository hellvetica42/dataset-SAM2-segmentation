import os
import json
import cv2
import argparse
import numpy as np
import time
from pathlib import Path
from tkinter import Tk, filedialog

# === SAM2 CHECKPOINT ===
# CHANGE THIS PATH TO YOUR ACTUAL CHECKPOINT LOCATION
CKPT_PATH = "/home/ubuntu/dataset-SAM2-segmentation/sam3/checkpoints/dataset-SAM2-segmentation/sam3/checkpoints"  
# =======================

from sam_utils import SAM3Runner
from video_utils import iter_pictures


# ---- COCO RLE helpers ----
def rle_encode_uncompressed(mask: np.ndarray):
    """COCO-style uncompressed RLE"""
    m = np.asfortranarray(mask.astype(np.uint8))
    H, W = m.shape
    r = m.reshape(-1, order="F")
    counts, run_val, run_len = [], 0, 0
    for v in r:
        if v == run_val:
            run_len += 1
        else:
            counts.append(run_len)
            run_val = int(v)
            run_len = 1
    counts.append(run_len)
    return {"counts": counts, "size": [int(H), int(W)]}


def rle_encode_coco(mask: np.ndarray):
    """Compressed RLE via pycocotools if available; else uncompressed."""
    try:
        from pycocotools import mask as maskUtils
        m = np.asfortranarray(mask.astype(np.uint8))
        rle = maskUtils.encode(m[:, :, None])[0]
        rle["counts"] = rle["counts"].decode("ascii")
        return rle
    except Exception:
        return rle_encode_uncompressed(mask)


def process_images(images_dir: Path, json_path: Path, out_json_path: Path,
                   category_filter: int | None):
    """Process all images with their annotations and add segmentation masks."""
    
    # Load the COCO JSON
    with open(json_path, "r") as f:
        coco = json.load(f)
    
    # Create mapping of annotation ID to annotation object
    ann_by_id = {ann["id"]: ann for ann in coco["annotations"]}

    # Initialize SAM3 once
    print(f"Initializing SAM3 model...")
    print(f"  Checkpoint: {CKPT_PATH}")
    
    if not Path(CKPT_PATH).exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}\nPlease update CKPT_PATH in the script.")
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cpu":
        print(" Warning: Running on CPU. This will be slow. Consider using a GPU.")
    
    sam = SAM3Runner(ckpt_path=CKPT_PATH, device=device)

    print("Starting segmentation...")

    processed_images = 0
    updated_anns = 0
    last_print_time = time.time()

    # Iterate through all images
    for frame_idx, frame_bgr, anns in iter_pictures(str(images_dir), str(json_path), 
                                                     category_filter=category_filter):
        
        # Extract bounding boxes from annotations
        boxes = [a["bbox"] for a in anns]
        
        if boxes:
            # Run SAM2 segmentation
            masks = sam.segment_boxes(frame_bgr, boxes)
            
            # Add segmentation to each annotation
            for ann_tmp, mask in zip(anns, masks):
                coco_ann = ann_by_id.get(ann_tmp["id"])
                if coco_ann is None:
                    continue
                coco_ann["segmentation"] = rle_encode_coco(mask)
                updated_anns += 1

        processed_images += 1
        
        # Progress logging every 50 images
        if processed_images % 50 == 0:
            now = time.time()
            elapsed = now - last_print_time
            last_print_time = now
            print(f"  Processed {processed_images} images ({updated_anns} annotations segmented) | {elapsed:.2f}s elapsed")
            
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json_path, "w") as f:
                json.dump(coco, f)
            print(f"  Saved intermediate results to: {out_json_path}")

    # Ensure output dir exists and save
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(coco, f)

    print(f"\n✓ Wrote: {out_json_path}")
    print(f"  Total images processed: {processed_images}")
    print(f"  Total annotations segmented: {updated_anns}")
    return True


# ---- GUI pickers ----
def pick_folder(title: str) -> Path:
    root = Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)


def pick_json_file(title: str) -> Path:
    root = Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=(
            ("JSON files", "*.json"),
            ("All files", "*.*")
        )
    )
    root.destroy()
    if not path:
        raise SystemExit("Cancelled.")
    return Path(path)


def main():
    ap = argparse.ArgumentParser(description="Add SAM3 segmentations to COCO JSON annotations for images.")
    ap.add_argument("--cat", type=int, default=None, help="Only segment this category_id (e.g. 1). None=all")
    ap.add_argument("--image_exts", default=".png,.jpg,.jpeg",
                    help="Comma-separated image extensions to match")
    args = ap.parse_args()

    print("Select the images folder…") 
    images_dir = pick_folder("Select images folder")

    print("Select the JSON annotation file…")
    json_path = pick_json_file("Select JSON annotation file")
    
    print(f"\nUsing:")
    print(f"  Images folder: {images_dir}")
    print(f"  JSON file: {json_path}")

    # Output directory - save next to the images folder
    out_dir = images_dir.parent / "labels_with_segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the same filename as input JSON
    out_json = out_dir / json_path.name.replace(".json", "_segmented.json")
    print(f"  Output will be saved to: {out_json}\n")

    # Process all images once
    process_images(
        images_dir=images_dir,
        json_path=json_path,
        out_json_path=out_json,
        category_filter=args.cat,
    )

    print(f"\n Done! Output: {out_json}")


if __name__ == "__main__":
    main()