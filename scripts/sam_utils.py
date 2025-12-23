import cv2
import torch
import numpy as np

"""
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
"""

from transformers import Sam3Processor, Sam3Model
from PIL import Image


def _boxes_xyxy_from_xywh(boxes_xywh):
    if not boxes_xywh:
        return np.empty((0, 4), dtype=np.float32)
    b = np.asarray(boxes_xywh, dtype=np.float32)
    b[:, 2] = b[:, 0] + b[:, 2]  # x2 
    b[:, 3] = b[:, 1] + b[:, 3]  # y2 
    return b[:, [0, 1, 2, 3]]


class SAM3Runner:
    """
    Initialize once; call .segment_boxes(frame_bgr, boxes_xywh) per frame.
    Returns one mask per input box.
    """
        
    def __init__(self, ckpt_path: str = "facebook/sam3", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        
        # Load model and processor from Transformers
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        

    def segment_boxes(self, frame_bgr: np.ndarray, boxes_xywh):
        """
        Returns:
            list of (H, W) uint8 masks, one per box (same order).
        """
        if not boxes_xywh:
            return []

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image (required by Transformers processor)
        pil_image = Image.fromarray(img_rgb)
        
        text = "tetra pack carton"
        boxes_xyxy = _boxes_xyxy_from_xywh(boxes_xywh)
        
        with torch.inference_mode():
            # Process image with BOTH text and box prompts
            inputs = self.processor(
                images=pil_image,
                text=text,
                input_boxes=[boxes_xyxy],  # List of boxes for the image
                input_boxes_labels=[[1] * len(boxes_xyxy)],  # 1 = positive prompt for each box
                return_tensors="pt"
            ).to(self.device)
            
            # Run model
            if self.device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            # Post-process to get masks
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
        
        # Extract masks from results
        pred_masks = results['masks']  # Binary masks at original size
        pred_boxes = results['boxes']  # xyxy format
        pred_scores = results['scores']
        
        # Match predicted masks to input boxes in order
        best_masks = self._match_masks_to_boxes_ordered(
            pred_masks, 
            pred_boxes, 
            boxes_xywh, 
            pred_scores,
            img_rgb.shape[:2]
        )

        if len(best_masks) != len(boxes_xywh):
            print(f"Warning: Expected {len(boxes_xywh)} masks, got {len(best_masks)}")

        return best_masks
    
    
    def _match_masks_to_boxes_ordered(self, pred_masks, pred_boxes, boxes_xywh, scores, img_shape):
        """
        Match predicted masks to input bounding boxes in the same order.
        Returns one mask per input box, maintaining order.
        """
        boxes_xyxy = _boxes_xyxy_from_xywh(boxes_xywh)
        matched_masks = []
        used_indices = set()
        
        for input_box in boxes_xyxy:
            best_iou = 0
            best_idx = -1
            
            # Find mask with highest IoU to this input box (that hasn't been used)
            for idx, (pred_box, score) in enumerate(zip(pred_boxes, scores)):
                if idx in used_indices:
                    continue
                    
                iou = self._compute_iou(input_box, pred_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # Convert mask to uint8
            if best_idx >= 0 and best_iou > 0.1:  # Minimum IoU threshold
                mask_uint8 = (pred_masks[best_idx].cpu().numpy() * 255).astype(np.uint8)
                used_indices.add(best_idx)
            else:
                # No matching mask found, create empty mask
                h, w = img_shape
                mask_uint8 = np.zeros((h, w), dtype=np.uint8)
                print(f"Warning: No mask found for box {input_box} (best IoU: {best_iou:.3f})")
            
            matched_masks.append(mask_uint8)
        
        return matched_masks
    
    
    def _compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes in xyxy format.
        box1, box2: [x_min, y_min, x_max, y_max]
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0