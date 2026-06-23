"""
ATVDCS – Main Pipeline Orchestrator
────────────────────────────────────────────────────────────────────────────────
Wires all 8 modules together into a single end-to-end processing pipeline.

Usage:
  # Single image
  python pipeline.py --image path/to/image.jpg

  # Directory of images
  python pipeline.py --dir path/to/images/ --camera CAM_01

  # Run demo with synthetic data (no real images needed)
  python pipeline.py --demo
"""

import argparse
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import yaml

from modules.preprocessing       import ImagePreprocessor, ImageMetadata
from modules.detection           import VehicleDetector, draw_detections
from modules.violation_classifier import ViolationClassifier
from modules.lpr                 import LicensePlateRecognizer
from modules.evidence            import EvidenceGenerator
from modules.analytics           import AnalyticsDB

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("atvdcs.pipeline")


# ── Pipeline class ────────────────────────────────────────────────────────────

class ATVDCSPipeline:
    """
    End-to-end ATVDCS pipeline.

    Modules initialised once, reused across every frame/image.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        logger.info("Initialising ATVDCS pipeline …")
        self.config_path = config_path

        # Module 01
        self.preprocessor   = ImagePreprocessor(config_path)
        # Module 02
        self.detector       = VehicleDetector(config_path)
        # Module 03 & 04
        self.classifier     = ViolationClassifier(config_path)
        # Module 05
        self.lpr            = LicensePlateRecognizer(config_path)
        # Module 06
        self.evidence_gen   = EvidenceGenerator(config_path)
        # Module 07
        self.analytics_db   = AnalyticsDB(config_path)

        logger.info("Pipeline ready ✓")

    # ── Core frame processor ──────────────────────────────────────────────────

    def process_image(
        self,
        image_path: "str | Path | np.ndarray",
        camera_id: str = "CAM_001",
        gps: Optional[tuple] = None,
        signal_red: bool = False,
        pipeline_run_id: Optional[str] = None,
    ) -> dict:
        """
        Run the full pipeline on a single image.

        Returns a summary dict with violation records and timing.
        """
        t_start = time.perf_counter()
        run_id  = pipeline_run_id or str(uuid.uuid4())[:8]

        # ── M01: Preprocessing ────────────────────────────────────────────
        meta = ImageMetadata(camera_id=camera_id, gps_lat=gps[0] if gps else None,
                             gps_lon=gps[1] if gps else None)
        preprocessed = self.preprocessor.process(image_path, meta)
        display_img  = preprocessed.image_display   # uint8 BGR, annotatable

        # ── M02: Detection ────────────────────────────────────────────────
        det_result = self.detector.detect(display_img)

        # ── M03 & M04: Violation Classification ──────────────────────────
        clf_result = self.classifier.classify(
            display_img, det_result,
            signal_red=signal_red,
            frame_id=run_id,
        )

        # ── M05: LPR ─────────────────────────────────────────────────────
        lpr_result = self.lpr.recognize(display_img)

        # ── M06: Evidence Generation ──────────────────────────────────────
        records = self.evidence_gen.generate(
            display_img, clf_result, lpr_result,
            camera_id=camera_id, gps=gps,
            pipeline_run_id=run_id,
        )

        # ── M07: Store in Analytics DB ────────────────────────────────────
        stored = self.analytics_db.insert_many(records)

        total_ms = (time.perf_counter() - t_start) * 1000
        best_plate = lpr_result.best()

        summary = {
            "run_id":           run_id,
            "camera_id":        camera_id,
            "violations_found": len(clf_result.violations),
            "auto_enforce":     len(clf_result.auto_enforce()),
            "human_review":     len(clf_result.needs_review()),
            "plate":            best_plate.plate_text if best_plate else None,
            "plate_valid":      best_plate.valid_format if best_plate else False,
            "records_saved":    stored,
            "latency_ms":       round(total_ms, 1),
            "detection_ms":     det_result.inference_time_ms,
            "classification_ms":clf_result.analysis_time_ms,
            "records":          records,
        }

        self._log_summary(summary)
        return summary

    def process_directory(
        self,
        dir_path: "str | Path",
        camera_id: str = "CAM_001",
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
        **kwargs,
    ) -> List[dict]:
        """Process all images in a directory."""
        dir_path = Path(dir_path)
        images = [p for p in dir_path.iterdir() if p.suffix.lower() in extensions]
        logger.info("Processing %d images from %s", len(images), dir_path)

        run_id = str(uuid.uuid4())[:8]
        results = []
        for img_path in sorted(images):
            result = self.process_image(img_path, camera_id=camera_id,
                                        pipeline_run_id=run_id, **kwargs)
            results.append(result)

        logger.info("Batch complete: %d images, %d total violations",
                    len(results), sum(r["violations_found"] for r in results))
        return results

    # ── Reports ───────────────────────────────────────────────────────────────

    def generate_daily_report(self, days: int = 1) -> str:
        logger.info("Generating %d-day report …", days)
        pdf = self.analytics_db.generate_pdf_report(days=days)
        self.analytics_db.plot_daily_trend(days=7)
        self.analytics_db.plot_violation_distribution(days=days)
        self.analytics_db.plot_confidence_histogram()
        return pdf

    def export_csv(self, **query_kwargs) -> str:
        return self.analytics_db.export_csv(**query_kwargs)

    # ── Logging helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _log_summary(s: dict):
        plate_str = s["plate"] or "—"
        valid_str = "(valid)" if s["plate_valid"] else "(unverified)"
        logger.info(
            "RUN %s | %d violation(s) | plate: %s %s | %.0f ms total",
            s["run_id"], s["violations_found"], plate_str, valid_str, s["latency_ms"],
        )
        for rec in s["records"]:
            logger.info(
                "  → %s | %s | conf=%.2f | disp=%s",
                rec.record_id, rec.violation_class,
                rec.violation_confidence, rec.disposition,
            )


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo(config_path: str = "config/config.yaml"):
    """
    Runs the full pipeline on a synthetic 640×640 test image.
    No real camera or image files required.
    """
    logger.info("=" * 55)
    logger.info("  ATVDCS DEMO — Synthetic Test Image")
    logger.info("=" * 55)

    pipeline = ATVDCSPipeline(config_path)

    # Build a synthetic scene: grey road + coloured vehicle blobs
    demo_img = np.full((640, 640, 3), 120, dtype=np.uint8)
    # Road markings
    cv2.rectangle(demo_img, (0, 400), (640, 410), (255, 255, 255), -1)   # stop line
    cv2.rectangle(demo_img, (290, 0), (350, 640), (255, 255, 0), 2)      # lane marking
    # Simulated car
    cv2.rectangle(demo_img, (50, 280), (280, 420), (30, 80, 200), -1)
    # Simulated motorcycle
    cv2.rectangle(demo_img, (380, 300), (500, 430), (10, 140, 80), -1)
    # Person on motorcycle
    cv2.rectangle(demo_img, (390, 230), (450, 310), (180, 100, 50), -1)
    # Fake plate area
    cv2.rectangle(demo_img, (80, 400), (230, 430), (255, 255, 255), -1)
    cv2.putText(demo_img, "MH12AB3456", (82, 422),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    result = pipeline.process_image(
        demo_img,
        camera_id="DEMO_CAM_01",
        gps=(19.076, 72.877),   # Mumbai
        signal_red=True,
    )

    # Print summary
    print("\n┌─────────────────────────────────────────┐")
    print("│         PIPELINE RESULT SUMMARY         │")
    print("├─────────────────────────────────────────┤")
    print(f"│  Violations detected : {result['violations_found']:<17} │")
    print(f"│  Auto-enforce        : {result['auto_enforce']:<17} │")
    print(f"│  Human review queue  : {result['human_review']:<17} │")
    print(f"│  Best plate          : {str(result['plate']):<17} │")
    print(f"│  Records saved       : {result['records_saved']:<17} │")
    print(f"│  Total latency       : {result['latency_ms']:<14.0f} ms │")
    print("└─────────────────────────────────────────┘\n")

    # Generate charts & report
    pipeline.analytics_db.plot_violation_distribution()
    logger.info("Demo complete. Check evidence/ and reports/ directories.")
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATVDCS Traffic Violation Pipeline")
    parser.add_argument("--image",    type=str, help="Path to a single image")
    parser.add_argument("--dir",      type=str, help="Path to image directory")
    parser.add_argument("--camera",   type=str, default="CAM_001")
    parser.add_argument("--red",      action="store_true", help="Simulate red light")
    parser.add_argument("--demo",     action="store_true", help="Run synthetic demo")
    parser.add_argument("--config",   type=str, default="config/config.yaml")
    parser.add_argument("--report",   action="store_true", help="Generate daily report")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.config)
    elif args.image:
        pl = ATVDCSPipeline(args.config)
        pl.process_image(args.image, camera_id=args.camera, signal_red=args.red)
        if args.report:
            pl.generate_daily_report()
    elif args.dir:
        pl = ATVDCSPipeline(args.config)
        pl.process_directory(args.dir, camera_id=args.camera, signal_red=args.red)
        if args.report:
            pl.generate_daily_report()
    else:
        parser.print_help()
