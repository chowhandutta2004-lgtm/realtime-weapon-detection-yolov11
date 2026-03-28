import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .config import settings

logger = logging.getLogger(__name__)

# Colors (BGR) for 4-class weapon taxonomy
LABEL_COLORS = {
    "handgun": (0, 0, 255),     # red
    "long_gun": (0, 0, 180),    # dark red
    "knife": (0, 140, 255),     # orange
    "explosive": (200, 0, 200), # magenta
}
DEFAULT_COLOR = (0, 255, 255)

# Maximum bounding box dimension as a fraction of the frame's largest side.
# A real handgun/knife seen by a webcam is a *small* object — if the box spans
# 40 %+ of the frame, it's almost certainly a false positive on a person/furniture.
MAX_BOX_DIM_RATIO = {
    "handgun": 0.35,
    "long_gun": 0.55,
    "knife": 0.45,      # knives can appear larger when held close to camera
    "explosive": 0.50,
}

# Knife class index — resolved once at model load time
_knife_class_idx: int | None = None


class WeaponDetector:
    def __init__(self):
        self.model = None

    def load(self):
        global _knife_class_idx
        model_path = Path(settings.MODEL_PATH)
        if model_path.exists():
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = YOLO(str(model_path))
        else:
            logger.warning(
                f"Fine-tuned model not found at {model_path}. "
                f"Using pretrained {settings.FALLBACK_MODEL}"
            )
            self.model = YOLO(settings.FALLBACK_MODEL)

        # Cache the knife class index for fast two-pass detection
        for idx, name in self.model.names.items():
            if name.lower() == "knife":
                _knife_class_idx = idx
                break
        logger.info(f"Knife class index: {_knife_class_idx}")

    def _filter_box(self, label: str, box_w: int, box_h: int,
                    frame_w: int, frame_h: int) -> bool:
        """Return True if the box should be REJECTED."""
        is_knife = label.lower() == "knife"
        box_area = box_w * box_h
        frame_area = frame_w * frame_h

        # Minimum area check (knives get a smaller minimum)
        min_area = settings.KNIFE_MIN_BOX_AREA if is_knife else settings.MIN_BOX_AREA
        if box_area < min_area:
            return True

        # Max frame ratio (knives exempt — they're thin so area ratio is low)
        if not is_knife and frame_area > 0 and box_area / frame_area > settings.MAX_BOX_FRAME_RATIO:
            return True

        # Max dimension ratio
        max_side = max(box_w, box_h)
        frame_max = max(frame_w, frame_h)
        dim_limit = MAX_BOX_DIM_RATIO.get(label.lower(), 0.50)
        if frame_max > 0 and max_side / frame_max > dim_limit:
            logger.debug(
                "Filtered %s: box %dx%d is %.0f%% of frame",
                label, box_w, box_h, max_side / frame_max * 100,
            )
            return True

        return False

    def _detect_pass(self, frame: np.ndarray, conf: float,
                     imgsz: int, use_tta: bool = False) -> list:
        """Run one inference pass and return raw results boxes."""
        return self.model(
            frame, conf=conf, iou=0.4, imgsz=imgsz, augment=use_tta, verbose=False,
        )[0]

    def _extract_detections(self, results, conf: float, knife_conf: float,
                            frame_w: int, frame_h: int,
                            offset_x: int = 0, offset_y: int = 0) -> list[dict]:
        """Extract filtered detections from a results object, applying offsets for cropped regions."""
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = results.names[int(box.cls[0])]
            score = float(box.conf[0])
            is_knife = label.lower() == "knife"

            min_conf = knife_conf if is_knife else conf
            if score < min_conf:
                continue

            # Apply offset for cropped regions
            bx1 = x1 + offset_x
            by1 = y1 + offset_y
            bx2 = x2 + offset_x
            by2 = y2 + offset_y

            if self._filter_box(label, bx2 - bx1, by2 - by1, frame_w, frame_h):
                continue

            detections.append({
                "label": label,
                "confidence": round(score, 3),
                "box": [int(bx1), int(by1), int(bx2), int(by2)],
            })
        return detections

    def _deduplicate(self, detections: list[dict], iou_thresh: float = 0.4) -> list[dict]:
        """Remove duplicate detections (same label, overlapping boxes) keeping highest confidence."""
        if len(detections) <= 1:
            return detections
        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        keep = []
        for det in detections:
            is_dup = False
            for kept in keep:
                if det["label"] != kept["label"]:
                    continue
                # Compute IoU
                ax1, ay1, ax2, ay2 = det["box"]
                bx1, by1, bx2, by2 = kept["box"]
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter
                if union > 0 and inter / union > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(det)
        return keep

    def detect_frame(
        self, frame: np.ndarray, confidence: float | None = None,
        use_tta: bool = False, deep_knife: bool = False,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Detect weapons in a frame.

        deep_knife=False (default) — fast single-pass, good for live webcam/CCTV.
        deep_knife=True — adds quadrant crops for better distant knife detection.
                          Use for image uploads and video file processing only.
        """
        conf = confidence if confidence is not None else settings.CONFIDENCE_THRESHOLD
        knife_conf = settings.KNIFE_CONFIDENCE_THRESHOLD
        h, w = frame.shape[:2]

        # Use 640 for live (fast), 1280 for deep mode (accurate)
        imgsz = 1280 if deep_knife else 640

        # Single pass at the lower knife threshold
        run_conf = min(conf, knife_conf)
        results = self._detect_pass(frame, run_conf, imgsz, use_tta)
        detections = self._extract_detections(results, conf, knife_conf, w, h)

        # Quadrant-based knife enhancement — only when deep_knife is on
        if deep_knife and _knife_class_idx is not None:
            crop_h, crop_w = h // 2, w // 2
            overlap = 60
            quadrants = [
                (0, 0, crop_w + overlap, crop_h + overlap),
                (max(0, crop_w - overlap), 0, w, crop_h + overlap),
                (0, max(0, crop_h - overlap), crop_w + overlap, h),
                (max(0, crop_w - overlap), max(0, crop_h - overlap), w, h),
            ]
            for (qx1, qy1, qx2, qy2) in quadrants:
                crop = frame[qy1:qy2, qx1:qx2]
                if crop.shape[0] < 32 or crop.shape[1] < 32:
                    continue
                crop_results = self._detect_pass(crop, knife_conf, 1280, False)
                crop_dets = self._extract_detections(
                    crop_results, conf, knife_conf, w, h,
                    offset_x=qx1, offset_y=qy1,
                )
                for d in crop_dets:
                    if d["label"].lower() == "knife":
                        detections.append(d)
            detections = self._deduplicate(detections)

        # Draw annotations (knives on top so they're always visible)
        annotated = frame.copy()
        detections.sort(key=lambda d: d["label"].lower() == "knife")

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            score = det["confidence"]
            color = LABEL_COLORS.get(label.lower(), DEFAULT_COLOR)

            thickness = 3 if label.lower() == "knife" else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            text = f"{label} {score:.2f}"
            (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th_text - 8), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        return annotated, detections


detector = WeaponDetector()
