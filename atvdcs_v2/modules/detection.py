"""
Module 02 – Vehicle & Road User Detection
────────────────────────────────────────────────────────────────────────────────
Runs YOLOv8 object detection on preprocessed images.

Detected categories:
  Group A : car, bus, truck/heavy vehicle, driver (frontal), traffic signal,
            stop line, number plate region
  Group B : motorcycle, passenger (lateral)
  Group C : auto-rickshaw, bicycle, pedestrian

Returns structured DetectionResult with bounding boxes, classes, and scores.
"""

import cv2
import numpy as np
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# COCO class indices used by YOLOv8 (subset relevant to traffic)
COCO_TRAFFIC_CLASSES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
}

# Map COCO labels to internal vehicle groups
VEHICLE_GROUPS = {
    "car":        "A",
    "bus":        "A",
    "truck":      "A",
    "motorcycle": "B",
    "bicycle":    "C",
    "person":     "C",
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 (pixel coords)
    class_id: int
    class_name: str
    confidence: float
    group: str = "C"                  # A | B | C per framework doc
    track_id: Optional[int] = None    # populated by tracker


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    image_shape: Tuple[int, int, int] = (640, 640, 3)   # H, W, C
    inference_time_ms: float = 0.0
    model_name: str = "yolov8n"

    # ── Convenience filters ──────────────────────────────────────────────────

    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name in ("car", "bus", "truck")]

    def motorcycles(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "motorcycle"]

    def persons(self) -> List[Detection]:
        return [d for d in self.detections if d.class_name == "person"]

    def by_class(self, name: str) -> List[Detection]:
        return [d for d in self.detections if d.class_name == name]

    def above_confidence(self, threshold: float) -> List[Detection]:
        return [d for d in self.detections if d.confidence >= threshold]


class VehicleDetector:
    """
    Wraps YOLOv8 (ultralytics) for traffic object detection.
    Falls back gracefully to a mock detector when ultralytics is not installed
    (useful for unit tests / CI without GPU).
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        det = cfg["detection"]

        self.model_name: str    = det["model"]
        self.conf_thresh: float = det["confidence_threshold"]
        self.iou_thresh: float  = det["iou_threshold"]
        self.device: str        = det["device"]
        self.class_filter: Dict[int, str] = {
            int(k): v for k, v in det["classes"].items()
        }

        self._model = self._load_model()

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        original_meta: Optional[Dict] = None,
    ) -> DetectionResult:
        """
        Run detection on a preprocessed display image (uint8 BGR).

        Parameters
        ----------
        image : uint8 BGR numpy array (640×640)
        original_meta : optional dict with camera/timestamp info

        Returns
        -------
        DetectionResult
        """
        import time
        t0 = time.perf_counter()
        detections = self._run_inference(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = DetectionResult(
            detections=detections,
            image_shape=image.shape,
            inference_time_ms=round(elapsed_ms, 2),
            model_name=self.model_name,
        )
        logger.debug(
            "Detected %d objects in %.1f ms", len(detections), elapsed_ms
        )
        return result

    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        return [self.detect(img) for img in images]

    # ── Internal ─────────────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info("Loading YOLOv8 model: %s on %s", self.model_name, self.device)
            model = YOLO(f"{self.model_name}.pt")
            return ("yolo", model)
        except ImportError:
            logger.warning(
                "ultralytics not installed — using mock detector. "
                "Install with: pip install ultralytics"
            )
            return ("mock", None)

    def _run_inference(self, image: np.ndarray) -> List[Detection]:
        kind, model = self._model

        if kind == "yolo":
            return self._yolo_inference(model, image)
        else:
            return self._mock_inference(image)

    def _yolo_inference(self, model, image: np.ndarray) -> List[Detection]:
        results = model.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
            classes=list(self.class_filter.keys()),
        )
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_filter.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        group=VEHICLE_GROUPS.get(cls_name, "C"),
                    )
                )
        return detections

    def _mock_inference(self, image: np.ndarray) -> List[Detection]:
        """Returns synthetic detections for testing without a real model."""
        h, w = image.shape[:2]
        return [
            Detection(
                bbox=(int(w * 0.1), int(h * 0.3), int(w * 0.4), int(h * 0.75)),
                class_id=2,
                class_name="car",
                confidence=0.91,
                group="A",
            ),
            Detection(
                bbox=(int(w * 0.55), int(h * 0.35), int(w * 0.80), int(h * 0.78)),
                class_id=3,
                class_name="motorcycle",
                confidence=0.87,
                group="B",
            ),
            Detection(
                bbox=(int(w * 0.60), int(h * 0.25), int(w * 0.75), int(h * 0.45)),
                class_id=0,
                class_name="person",
                confidence=0.82,
                group="C",
            ),
        ]


# ── Visualisation helper ─────────────────────────────────────────────────────

def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and labels onto a BGR image copy."""
    CLASS_COLORS = {
        "car":        (0, 255, 0),
        "bus":        (0, 200, 100),
        "truck":      (0, 180, 80),
        "motorcycle": (255, 165, 0),
        "bicycle":    (200, 200, 0),
        "person":     (255, 0, 255),
    }
    vis = image.copy()
    for det in result.detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, (200, 200, 200))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = det.class_name
        if show_confidence:
            label += f" {det.confidence:.2f}"
        cv2.putText(vis, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # Inference time watermark
    cv2.putText(
        vis,
        f"{result.model_name}  {result.inference_time_ms:.0f}ms",
        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )
    return vis
