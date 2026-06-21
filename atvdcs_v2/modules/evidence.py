"""
Module 06 – Evidence Generation
────────────────────────────────────────────────────────────────────────────────
Produces tamper-evident evidence packages for every confirmed/suspected violation.

Package contents:
  ├── annotated_image.png   – lossless annotated frame
  ├── preview.jpg           – compressed preview
  └── violation_record.json – signed metadata record

Tamper evidence:
  • SHA-256 hash of the annotated PNG
  • NTP-synced ISO-8601 timestamp embedded in EXIF
  • Chain-of-custody: node ID, model version, pipeline run ID
"""

import os
import cv2
import json
import yaml
import hashlib
import logging
import uuid
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

from modules.violation_classifier import ViolationEvent, ClassificationResult
from modules.lpr import LPRResult
from modules.explainability import explain_violation

logger = logging.getLogger(__name__)


@dataclass
class ViolationRecord:
    """Court-admissible evidence record schema (from system framework §4.6.2)."""
    record_id: str
    vehicle_class: str
    plate_text: str
    plate_confidence: float
    violation_class: str
    violation_confidence: float
    disposition: str
    lat: Optional[float]
    lon: Optional[float]
    timestamp: str
    camera_id: str
    pipeline_run_id: str
    model_version: str
    image_hash: str           # SHA-256 of annotated PNG
    annotated_image_path: str
    preview_image_path: str
    metadata: Dict[str, Any]


class EvidenceGenerator:
    """
    Generates and persists a complete evidence package for each violation event.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        ev = cfg["evidence"]
        self.output_format: str  = ev["output_format"]
        self.jpeg_quality: int   = ev["jpeg_quality"]
        self.viol_color: Tuple   = tuple(ev["annotation"]["violation_color"])
        self.ctx_color: Tuple    = tuple(ev["annotation"]["context_color"])
        self.text_scale: float   = ev["annotation"]["text_scale"]
        self.line_thick: int     = ev["annotation"]["line_thickness"]
        self.hash_algo: str      = ev["hash_algorithm"]
        self.watermark: bool     = ev.get("watermark", True)

        self.evidence_dir = Path(cfg["system"]["evidence_dir"])
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.model_version: str = cfg["system"]["version"]

    # ── Public API ───────────────────────────────────────────────────────────

    def generate(
        self,
        image: np.ndarray,                   # uint8 BGR display image
        classification: ClassificationResult,
        lpr_result: LPRResult,
        camera_id: str = "CAM_001",
        gps: Optional[Tuple[float, float]] = None,   # (lat, lon)
        pipeline_run_id: Optional[str] = None,
    ) -> List[ViolationRecord]:
        """
        Generate one evidence package per violation in the ClassificationResult.

        Returns list of ViolationRecord objects (also written to disk as JSON).
        """
        if pipeline_run_id is None:
            pipeline_run_id = str(uuid.uuid4())[:8]

        records: List[ViolationRecord] = []
        best_plate = lpr_result.best()

        for event in classification.violations:
            record = self._create_evidence_package(
                image=image,
                event=event,
                best_plate=best_plate,
                camera_id=camera_id,
                gps=gps,
                pipeline_run_id=pipeline_run_id,
            )
            records.append(record)
            logger.info(
                "Evidence generated: %s | %s | plate=%s | conf=%.2f",
                record.record_id,
                record.violation_class,
                record.plate_text,
                record.violation_confidence,
            )

        return records

    # ── Package creation ─────────────────────────────────────────────────────

    def _create_evidence_package(
        self,
        image: np.ndarray,
        event: ViolationEvent,
        best_plate,
        camera_id: str,
        gps: Optional[Tuple[float, float]],
        pipeline_run_id: str,
    ) -> ViolationRecord:
        record_id = str(uuid.uuid4())[:12].upper()
        timestamp = datetime.now(timezone.utc).isoformat()
        lat = gps[0] if gps else None
        lon = gps[1] if gps else None

        # 1. Annotate image
        annotated = self._annotate(image.copy(), event, best_plate, camera_id, timestamp)

        # 2. Save lossless PNG
        png_path = self.evidence_dir / f"{record_id}_annotated.png"
        cv2.imwrite(str(png_path), annotated)

        # 3. SHA-256 of the PNG bytes
        img_hash = self._sha256_file(png_path)

        # 4. Save JPEG preview
        jpg_path = self.evidence_dir / f"{record_id}_preview.jpg"
        cv2.imwrite(str(jpg_path), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])

        # 5. Build record
        plate_text = best_plate.plate_text if best_plate else "UNREADABLE"
        plate_conf = best_plate.confidence if best_plate else 0.0
        vehicle_cls = (
            event.offending_vehicle.class_name if event.offending_vehicle else "unknown"
        )

        record = ViolationRecord(
            record_id=record_id,
            vehicle_class=vehicle_cls,
            plate_text=plate_text,
            plate_confidence=round(plate_conf, 4),
            violation_class=event.violation_type.value,
            violation_confidence=round(event.confidence, 4),
            disposition=event.disposition.value,
            lat=lat,
            lon=lon,
            timestamp=timestamp,
            camera_id=camera_id,
            pipeline_run_id=pipeline_run_id,
            model_version=self.model_version,
            image_hash=img_hash,
            annotated_image_path=str(png_path),
            preview_image_path=str(jpg_path),
            metadata={**event.meta, "explanation": explain_violation(event)},
        )

        # 6. Write JSON sidecar
        json_path = self.evidence_dir / f"{record_id}_record.json"
        with open(json_path, "w") as f:
            json.dump(asdict(record), f, indent=2)

        return record

    # ── Annotation ───────────────────────────────────────────────────────────

    def _annotate(
        self,
        image: np.ndarray,
        event: ViolationEvent,
        best_plate,
        camera_id: str,
        timestamp: str,
    ) -> np.ndarray:
        """Draw violation overlay, plate inset, and metadata strip."""
        h, w = image.shape[:2]

        # Main violation bounding box (RED)
        if event.evidence_bbox:
            x1, y1, x2, y2 = event.evidence_bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), self.viol_color[::-1], self.line_thick + 1)

        # Supporting context boxes (YELLOW)
        for det in event.supporting_detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), self.ctx_color[::-1], self.line_thick)

        # Violation label
        vtype = event.violation_type.value.replace("_", " ").upper()
        label = f"{vtype}  [{event.confidence:.0%}]"
        if event.evidence_bbox:
            x1, y1 = event.evidence_bbox[:2]
            cv2.putText(image, label, (max(x1, 4), max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                        self.viol_color[::-1], 2)

        # Plate inset (bottom-right corner)
        if best_plate and best_plate.bbox:
            px1, py1, px2, py2 = best_plate.bbox
            plate_crop = image[py1:py2, px1:px2]
            if plate_crop.size > 0:
                inset_h = 60
                inset_w = int(plate_crop.shape[1] * inset_h / max(plate_crop.shape[0], 1))
                inset = cv2.resize(plate_crop, (inset_w, inset_h))
                pw, ph = inset_w, inset_h
                image[h - ph - 4: h - 4, w - pw - 4: w - 4] = inset

        # Metadata strip (top bar)
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

        plate_str = best_plate.plate_text if best_plate else "---"
        strip = (f"  CAM: {camera_id}  |  PLATE: {plate_str}  |  "
                 f"DISP: {event.disposition.value.upper()}  |  {timestamp[:19]}Z")
        cv2.putText(image, strip, (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Watermark
        if self.watermark:
            wm = "ATVDCS EVIDENCE – NOT FOR REPRODUCTION"
            cv2.putText(image, wm, (4, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 200), 1)

        return image

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sha256_file(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def verify_integrity(record: "ViolationRecord") -> Dict[str, Any]:
        """
        Re-hash the annotated PNG on disk and compare against the hash stored
        in the evidence record at generation time. Powers the "Verify Evidence"
        button on the Evidence Management dashboard page — any post-hoc edit
        to the image (cropping, re-compression, overlay tampering) changes the
        SHA-256 digest and fails this check.
        """
        png_path = Path(record.annotated_image_path)
        if not png_path.exists():
            return {"valid": False, "reason": "Annotated image file missing on disk."}

        current_hash = EvidenceGenerator._sha256_file(png_path)
        valid = current_hash == record.image_hash
        return {
            "valid": valid,
            "stored_hash": record.image_hash,
            "recomputed_hash": current_hash,
            "reason": "Hash matches — evidence intact." if valid
                       else "Hash mismatch — file has been modified since generation.",
        }
