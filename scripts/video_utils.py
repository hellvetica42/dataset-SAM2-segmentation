import cv2
import json
from pathlib import Path
from collections import defaultdict

def read_coco_json(json_path):
    """
    Loads COCO-style detections JSON and groups annotations by image_id.
    Returns:
        images_sorted: list of image dicts sorted by file_name
        anns_per_img: dict mapping image_id -> list of annotations
    """
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    images_sorted = sorted(images, key=lambda im: im["file_name"])
    anns_per_img = defaultdict(list)
    for ann in annotations:
        anns_per_img[ann["image_id"]].append(ann)

    return images_sorted, anns_per_img


def iter_pictures(images_folder, json_path=None, category_filter=None):
    """
    Opens images from folder and COCO JSON and yields them one by one.
    Yields:
        (frame_idx, frame, anns) where:
            frame_idx (int): index of the image
            frame (np.ndarray): BGR image from cv2
            anns (list): filtered annotations for this image (empty if no JSON)
    """

    images_sorted, anns_per_img = ([], {})
    if json_path:
        images_sorted, anns_per_img = read_coco_json(json_path)
        
    # Normalize category_filter to list
    if category_filter is not None:
        if isinstance(category_filter, int):
            category_filter = [category_filter]

    for frame_idx, img_info in enumerate(images_sorted):
        img_path = Path(images_folder) / img_info["file_name"]

        # Load image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: Could not read image: {img_path}")
            continue

        # Get annotations for this image
        img_id = img_info["id"]
        anns = anns_per_img.get(img_id, [])

        # Apply category_id filtering
        if category_filter is not None:
            anns = [a for a in anns if a.get("category_id") in category_filter]

        yield frame_idx, frame, anns