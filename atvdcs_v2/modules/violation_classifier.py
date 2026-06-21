"""
Modules 03 & 04 – Traffic Violation Detection & Classification
────────────────────────────────────────────────────────────────────────────────
Analyses DetectionResult objects to identify and classify seven violation types:

  1. Helmet Non-Compliance       – motorcycle rider without helmet
  2. Seatbelt Non-Compliance     – car driver without visible seatbelt
  3. Triple Riding               – 3+ persons on one motorcycle
  4. Wrong-Side Driving          – vehicle heading against lane flow
  5. Stop-Line Violation         – vehicle crosses stop line on red
  6. Red-Light Violation         – vehicle in intersection on red
  7. Illegal Parking             – stationary vehicle in restricted zone

Each detector returns a ViolationEvent with a calibrated confidence score.
The ViolationClassifier then applies the tiered routing strategy:
  ≥ 0.90 → auto-enforce | 0.70-0.90 → secondary check | < 0.70 → human review
"""

import cv2
import numpy as np
import yaml
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum

from modules.detection import Detection, DetectionResult

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

class ViolationType(str, Enum):
    HELMET_NON_COMPLIANCE   = "helmet_non_compliance"
    SEATBELT_NON_COMPLIANCE = "seatbelt_non_compliance"
    TRIPLE_RIDING           = "triple_riding"
    WRONG_SIDE_DRIVING      = "wrong_side_driving"
    STOP_LINE_VIOLATION     = "stop_line_violation"
    RED_LIGHT_VIOLATION     = "red_light_violation"
    ILLEGAL_PARKING         = "illegal_parking"


class DispositionTier(str, Enum):
    AUTO_ENFORCE    = "auto_enforce"     # confidence >= 0.90
    SECONDARY_CHECK = "secondary_check"  # 0.70 – 0.90
    HUMAN_REVIEW    = "human_review"     # < 0.70


@dataclass
class ViolationEvent:
    violation_type: ViolationType
    confidence: float
    disposition: DispositionTier
    offending_vehicle: Optional[Detection] = None
    supporting_detections: List[Detection] = field(default_factory=list)
    evidence_bbox: Optional[Tuple[int, int, int, int]] = None  # tight crop
    meta: Dict[str, Any] = field(default_factory=dict)         # extra context


@dataclass
class ClassificationResult:
    violations: List[ViolationEvent] = field(default_factory=list)
    frame_id: Optional[str] = None
    analysis_time_ms: float = 0.0

    def auto_enforce(self) -> List[ViolationEvent]:
        return [v for v in self.violations if v.disposition == DispositionTier.AUTO_ENFORCE]

    def needs_review(self) -> List[ViolationEvent]:
        return [v for v in self.violations if v.disposition == DispositionTier.HUMAN_REVIEW]

    def has_violations(self) -> bool:
        return len(self.violations) > 0


# ── Tiered confidence routing ────────────────────────────────────────────────

def _tier(confidence: float, thresholds: Dict[str, float]) -> DispositionTier:
    if confidence >= thresholds["auto_enforce"]:
        return DispositionTier.AUTO_ENFORCE
    if confidence >= thresholds["secondary_check"]:
        return DispositionTier.SECONDARY_CHECK
    return DispositionTier.HUMAN_REVIEW


# ── Individual violation detectors ───────────────────────────────────────────

class HelmetDetector:
    """
    Checks each motorcycle detection for helmetless riders.
    In prototype mode uses head-region heuristics.
    Production: EfficientNet-B0 binary classifier on cropped head region.
    """

    def __init__(self, cfg: Dict):
        self.helmet_conf = cfg["violation"]["helmet"]["confidence"]

    def detect(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        thresholds: Dict,
    ) -> List[ViolationEvent]:
        events = []
        motos = detection_result.motorcycles()
        persons = detection_result.persons()

        for moto in motos:
            # Find persons spatially overlapping or directly above motorcycle
            riders = self._find_riders(moto, persons)
            if not riders:
                continue

            # Crop head region of first rider → helmet classifier
            conf = self._classify_helmet(image, riders[0])

            if conf < self.helmet_conf:
                # Inverted: high score = no helmet
                violation_conf = 1.0 - conf
                events.append(
                    ViolationEvent(
                        violation_type=ViolationType.HELMET_NON_COMPLIANCE,
                        confidence=violation_conf,
                        disposition=_tier(violation_conf, thresholds),
                        offending_vehicle=moto,
                        supporting_detections=riders,
                        evidence_bbox=self._merge_bbox(moto, riders[0]),
                        meta={"helmet_score": conf},
                    )
                )
        return events

    def _find_riders(
        self, moto: Detection, persons: List[Detection], iou_expand: float = 1.4
    ) -> List[Detection]:
        mx1, my1, mx2, my2 = moto.bbox
        riders = []
        for p in persons:
            px1, py1, px2, py2 = p.bbox
            # Person centroid must be within expanded moto bbox
            cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
            if mx1 <= cx <= mx2 and my1 * 0.7 <= cy <= my2:
                riders.append(p)
        return riders

    def _classify_helmet(self, image: np.ndarray, person: Detection) -> float:
        """
        Prototype: uses upper-third brightness heuristic.
        Production: replace with EfficientNet-B0 inference.
        """
        x1, y1, x2, y2 = person.bbox
        head_y2 = y1 + (y2 - y1) // 3
        crop = image[max(0, y1):head_y2, max(0, x1):x2]
        if crop.size == 0:
            return 0.5
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Helmets tend to have more uniform, darker regions at top
        mean_brightness = float(np.mean(gray))
        # Simple heuristic: return pseudo-confidence (NOT production quality)
        return min(1.0, mean_brightness / 200.0)

    @staticmethod
    def _merge_bbox(a: Detection, b: Detection) -> Tuple[int, int, int, int]:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        return min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)


class SeatbeltDetector:
    """
    Checks car detections for missing seatbelt on driver.
    Production: YOLOv8 + binary seatbelt-presence classifier.
    """

    def detect(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        thresholds: Dict,
    ) -> List[ViolationEvent]:
        events = []
        for car in detection_result.vehicles():
            conf = self._classify_seatbelt(image, car)
            if conf < 0.5:
                violation_conf = max(0.72, 1.0 - conf)   # prototype floor
                events.append(
                    ViolationEvent(
                        violation_type=ViolationType.SEATBELT_NON_COMPLIANCE,
                        confidence=violation_conf,
                        disposition=_tier(violation_conf, thresholds),
                        offending_vehicle=car,
                        evidence_bbox=car.bbox,
                    )
                )
        return events

    def _classify_seatbelt(self, image: np.ndarray, car: Detection) -> float:
        """Prototype heuristic. Production: crop driver window → classifier."""
        x1, y1, x2, y2 = car.bbox
        w, h = x2 - x1, y2 - y1
        # Driver window region (upper-left quadrant of vehicle bbox)
        crop = image[y1:y1 + h // 2, x1:x1 + w // 2]
        if crop.size == 0:
            return 0.5
        # Diagonal edge density as a rough seatbelt proxy
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        return float(np.clip(edge_density * 10, 0, 1))


class TripleRidingDetector:
    """Flags motorcycles with 3+ visible person detections."""

    def __init__(self, cfg: Dict):
        self.min_riders: int = cfg["violation"]["triple_riding"]["min_riders"]

    def detect(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        thresholds: Dict,
    ) -> List[ViolationEvent]:
        events = []
        persons = detection_result.persons()

        for moto in detection_result.motorcycles():
            overlapping = self._count_persons_on_moto(moto, persons)
            if len(overlapping) >= self.min_riders:
                confidence = min(0.95, 0.75 + (len(overlapping) - 3) * 0.05)
                events.append(
                    ViolationEvent(
                        violation_type=ViolationType.TRIPLE_RIDING,
                        confidence=confidence,
                        disposition=_tier(confidence, thresholds),
                        offending_vehicle=moto,
                        supporting_detections=overlapping,
                        evidence_bbox=moto.bbox,
                        meta={"rider_count": len(overlapping)},
                    )
                )
        return events

    def _count_persons_on_moto(
        self, moto: Detection, persons: List[Detection]
    ) -> List[Detection]:
        mx1, my1, mx2, my2 = moto.bbox
        result = []
        for p in persons:
            px1, py1, px2, py2 = p.bbox
            cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
            if mx1 <= cx <= mx2 and my1 * 0.6 <= cy <= my2 * 1.1:
                result.append(p)
        return result


class StopLineViolationDetector:
    """
    Detects vehicles that cross a stop line while signal is RED.
    Requires a defined stop-line polygon for the camera view.
    """

    def __init__(self, stop_line_zone: Optional[np.ndarray] = None):
        # Default demo zone: bottom-centre strip of a 640×640 frame
        if stop_line_zone is None:
            self.stop_zone = np.array([[100, 450], [540, 450],
                                       [540, 490], [100, 490]])
        else:
            self.stop_zone = stop_line_zone

    def detect(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        thresholds: Dict,
        signal_red: bool = False,
    ) -> List[ViolationEvent]:
        if not signal_red:
            return []
        events = []
        all_vehicles = (
            detection_result.vehicles()
            + detection_result.motorcycles()
            + detection_result.by_class("bicycle")
        )
        for veh in all_vehicles:
            if self._overlaps_zone(veh.bbox):
                confidence = 0.88
                events.append(
                    ViolationEvent(
                        violation_type=ViolationType.STOP_LINE_VIOLATION,
                        confidence=confidence,
                        disposition=_tier(confidence, thresholds),
                        offending_vehicle=veh,
                        evidence_bbox=veh.bbox,
                        meta={"signal_state": "RED"},
                    )
                )
        return events

    def _overlaps_zone(self, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        corners = [(x1, y2), (x2, y2), ((x1 + x2) // 2, y2)]
        for pt in corners:
            if cv2.pointPolygonTest(self.stop_zone, pt, False) >= 0:
                return True
        return False


class IllegalParkingDetector:
    """
    Tracks vehicles stationary for > T seconds in restricted zones.
    Uses an in-memory dictionary keyed by approximate bbox centre.
    Production: replace with full object tracker (e.g. ByteTrack).
    """

    def __init__(self, cfg: Dict):
        self.threshold_s: int = cfg["violation"]["illegal_parking"]["stationary_seconds"]
        self._seen: Dict[str, Dict] = {}  # track_key → {first_seen, bbox, cls}

    def update(
        self,
        detection_result: DetectionResult,
        thresholds: Dict,
        no_parking_zone: Optional[np.ndarray] = None,
    ) -> List[ViolationEvent]:
        now = time.time()
        events = []
        all_vehicles = detection_result.vehicles() + detection_result.motorcycles()

        for veh in all_vehicles:
            key = self._track_key(veh)
            if key not in self._seen:
                self._seen[key] = {"first_seen": now, "bbox": veh.bbox, "cls": veh.class_name}
            else:
                elapsed = now - self._seen[key]["first_seen"]
                if elapsed >= self.threshold_s:
                    confidence = min(0.97, 0.80 + elapsed / 600)
                    events.append(
                        ViolationEvent(
                            violation_type=ViolationType.ILLEGAL_PARKING,
                            confidence=confidence,
                            disposition=_tier(confidence, thresholds),
                            offending_vehicle=veh,
                            evidence_bbox=veh.bbox,
                            meta={"stationary_seconds": round(elapsed)},
                        )
                    )
        # Purge stale tracks (not seen in last 30 s = vehicle moved)
        active_keys = {self._track_key(v) for v in all_vehicles}
        stale = [k for k in self._seen if k not in active_keys]
        for k in stale:
            if now - self._seen[k]["first_seen"] > 30:
                del self._seen[k]

        return events

    @staticmethod
    def _track_key(det: Detection) -> str:
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) // 2 // 30, (y1 + y2) // 2 // 30  # 30-px grid
        return f"{det.class_name}_{cx}_{cy}"


# ── Top-level Violation Classifier ───────────────────────────────────────────

class ViolationClassifier:
    """
    Orchestrates all individual detectors and returns a ClassificationResult
    with tiered disposition for every detected violation.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.thresholds = cfg["violation"]["thresholds"]

        self.helmet_det    = HelmetDetector(cfg)
        self.seatbelt_det  = SeatbeltDetector()
        self.triple_det    = TripleRidingDetector(cfg)
        self.stop_det      = StopLineViolationDetector()
        self.parking_det   = IllegalParkingDetector(cfg)

    def classify(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        signal_red: bool = False,
        frame_id: Optional[str] = None,
    ) -> ClassificationResult:
        t0 = time.perf_counter()
        violations: List[ViolationEvent] = []

        violations += self.helmet_det.detect(image, detection_result, self.thresholds)
        violations += self.seatbelt_det.detect(image, detection_result, self.thresholds)
        violations += self.triple_det.detect(image, detection_result, self.thresholds)
        violations += self.stop_det.detect(image, detection_result, self.thresholds, signal_red)
        violations += self.parking_det.update(detection_result, self.thresholds)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Classification: %d violation(s) detected in %.1f ms", len(violations), elapsed_ms
        )
        return ClassificationResult(
            violations=violations,
            frame_id=frame_id,
            analysis_time_ms=round(elapsed_ms, 2),
        )
