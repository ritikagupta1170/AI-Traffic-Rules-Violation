"""
ATVDCS – Test Suite
────────────────────────────────────────────────────────────────────────────────
Runs without any real images or model weights (mock fallbacks kick in).

  pytest tests/test_pipeline.py -v
"""

import sys
import os
import numpy as np
import pytest

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

CONFIG = "config/config.yaml"

# ── Synthetic test image ──────────────────────────────────────────────────────

def make_test_image(h: int = 640, w: int = 640) -> np.ndarray:
    """640×640 BGR synthetic road scene."""
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    import cv2
    cv2.rectangle(img, (50, 280), (280, 420), (30, 80, 200), -1)    # car
    cv2.rectangle(img, (380, 300), (500, 430), (10, 140, 80), -1)   # moto
    cv2.rectangle(img, (390, 230), (450, 310), (180, 100, 50), -1)  # person
    cv2.rectangle(img, (80, 400), (230, 430), (255, 255, 255), -1)  # plate
    return img


# ── Module 01: Preprocessing ─────────────────────────────────────────────────

class TestPreprocessing:
    def setup_method(self):
        from modules.preprocessing import ImagePreprocessor
        self.pp = ImagePreprocessor(CONFIG)
        self.img = make_test_image()

    def test_output_shape(self):
        result = self.pp.process(self.img)
        assert result.image.shape == (640, 640, 3), "Normalised image wrong shape"

    def test_display_image_uint8(self):
        result = self.pp.process(self.img)
        assert result.image_display.dtype == np.uint8

    def test_normalised_range(self):
        result = self.pp.process(self.img)
        # After ImageNet normalisation values can be negative
        assert result.image.dtype == np.float32

    def test_metadata_populated(self):
        result = self.pp.process(self.img)
        assert result.metadata.original_size == (640, 640)

    def test_enhancement_log(self):
        result = self.pp.process(self.img)
        assert "clahe" in result.enhancement_log


# ── Module 02: Detection ──────────────────────────────────────────────────────

class TestDetection:
    def setup_method(self):
        from modules.detection import VehicleDetector
        self.det = VehicleDetector(CONFIG)
        self.img = make_test_image()

    def test_returns_result_object(self):
        from modules.detection import DetectionResult
        result = self.det.detect(self.img)
        assert isinstance(result, DetectionResult)

    def test_mock_detections_present(self):
        result = self.det.detect(self.img)
        assert len(result.detections) > 0, "Mock detector should return detections"

    def test_detection_fields(self):
        result = self.det.detect(self.img)
        d = result.detections[0]
        assert len(d.bbox) == 4
        assert 0.0 <= d.confidence <= 1.0
        assert d.class_name in ("car", "motorcycle", "bus", "truck",
                                "person", "bicycle")

    def test_filter_helpers(self):
        result = self.det.detect(self.img)
        vehicles = result.vehicles()
        motos = result.motorcycles()
        persons = result.persons()
        total = len(vehicles) + len(motos) + len(persons)
        assert total <= len(result.detections)


# ── Module 03 & 04: Violation Classification ─────────────────────────────────

class TestViolationClassifier:
    def setup_method(self):
        from modules.detection import VehicleDetector
        from modules.violation_classifier import ViolationClassifier
        self.det = VehicleDetector(CONFIG)
        self.clf = ViolationClassifier(CONFIG)
        self.img = make_test_image()

    def test_classify_returns_result(self):
        from modules.violation_classifier import ClassificationResult
        det = self.det.detect(self.img)
        result = self.clf.classify(self.img, det, signal_red=True)
        assert isinstance(result, ClassificationResult)

    def test_confidence_in_range(self):
        det = self.det.detect(self.img)
        result = self.clf.classify(self.img, det)
        for v in result.violations:
            assert 0.0 <= v.confidence <= 1.0, f"Confidence out of range: {v.confidence}"

    def test_disposition_valid(self):
        from modules.violation_classifier import DispositionTier
        det = self.det.detect(self.img)
        result = self.clf.classify(self.img, det, signal_red=True)
        valid = {d.value for d in DispositionTier}
        for v in result.violations:
            assert v.disposition.value in valid

    def test_auto_enforce_filter(self):
        det = self.det.detect(self.img)
        result = self.clf.classify(self.img, det)
        ae = result.auto_enforce()
        assert all(v.confidence >= 0.90 for v in ae)


# ── Module 05: LPR ────────────────────────────────────────────────────────────

class TestLPR:
    def setup_method(self):
        from modules.lpr import LicensePlateRecognizer
        self.lpr = LicensePlateRecognizer(CONFIG)
        self.img = make_test_image()

    def test_returns_lpr_result(self):
        from modules.lpr import LPRResult
        result = self.lpr.recognize(self.img)
        assert isinstance(result, LPRResult)

    def test_clean_text_no_spaces(self):
        from modules.lpr import LicensePlateRecognizer
        lpr = LicensePlateRecognizer.__new__(LicensePlateRecognizer)
        lpr.plate_re = __import__("re").compile(r"^[A-Z]{2}[-]?\d{1,2}[-]?[A-Z]{1,3}[-]?\d{4}$")
        cleaned = lpr._clean_text("MH 12 AB 3456")
        assert " " not in cleaned
        assert "-" not in cleaned

    def test_ocr_correction(self):
        from modules.lpr import LicensePlateRecognizer
        lpr = LicensePlateRecognizer.__new__(LicensePlateRecognizer)
        lpr.plate_re = __import__("re").compile(r"^[A-Z]{2}[-]?\d{1,2}[-]?[A-Z]{1,3}[-]?\d{4}$")
        # O → 0 in digit positions
        cleaned = lpr._clean_text("MH12ABOO56")  # OO in digit zone
        assert "O" not in cleaned[-4:], "OCR correction should replace O with 0 in digit zone"


# ── Module 06: Evidence ───────────────────────────────────────────────────────

class TestEvidence:
    def setup_method(self):
        from modules.evidence import EvidenceGenerator
        from modules.detection import VehicleDetector
        from modules.violation_classifier import ViolationClassifier
        from modules.lpr import LicensePlateRecognizer
        self.ev_gen = EvidenceGenerator(CONFIG)
        self.det    = VehicleDetector(CONFIG)
        self.clf    = ViolationClassifier(CONFIG)
        self.lpr    = LicensePlateRecognizer(CONFIG)
        self.img    = make_test_image()

    def test_evidence_files_created(self):
        import os
        det    = self.det.detect(self.img)
        clf    = self.clf.classify(self.img, det, signal_red=True)
        lpr    = self.lpr.recognize(self.img)
        records = self.ev_gen.generate(self.img, clf, lpr)
        for rec in records:
            assert os.path.exists(rec.annotated_image_path), "PNG not created"
            assert os.path.exists(rec.preview_image_path),   "JPG not created"

    def test_sha256_hash_format(self):
        det    = self.det.detect(self.img)
        clf    = self.clf.classify(self.img, det, signal_red=True)
        lpr    = self.lpr.recognize(self.img)
        records = self.ev_gen.generate(self.img, clf, lpr)
        for rec in records:
            assert len(rec.image_hash) == 64, "SHA-256 should be 64 hex chars"


# ── Module 07: Analytics ──────────────────────────────────────────────────────

class TestAnalytics:
    def setup_method(self):
        from modules.analytics import AnalyticsDB
        self.db = AnalyticsDB(CONFIG)

    def test_summary_empty_db(self):
        stats = self.db.summary_stats(days=1)
        assert "total" in stats

    def test_query_returns_dataframe(self):
        import pandas as pd
        df = self.db.query(limit=10)
        assert isinstance(df, pd.DataFrame)


# ── Module 08: Evaluation ─────────────────────────────────────────────────────

class TestEvaluation:
    def setup_method(self):
        from modules.evaluation import PipelineEvaluator
        self.ev = PipelineEvaluator(CONFIG)

    def _make_predictions(self, n: int = 20):
        import random
        classes = ["helmet_non_compliance", "seatbelt_non_compliance",
                   "triple_riding", "none"]
        return [
            {
                "plate_pred": "MH12AB3456",
                "plate_gt":   "MH12AB3456",
                "violation_pred": random.choice(classes),
                "violation_gt":   classes[i % len(classes)],
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "latency_ms": random.uniform(80, 350),
            }
            for i in range(n)
        ]

    def test_report_fields(self):
        from modules.evaluation import EvalReport
        preds = self._make_predictions()
        report = self.ev.evaluate_from_predictions(preds)
        assert isinstance(report, EvalReport)
        assert 0.0 <= report.violation.f1 <= 1.0
        assert 0.0 <= report.lpr.plate_accuracy <= 1.0
        assert report.latency.p50_ms > 0

    def test_targets_checked(self):
        preds = self._make_predictions()
        report = self.ev.evaluate_from_predictions(preds)
        assert isinstance(report.targets_met, dict)
        assert "f1_violation" in report.targets_met


# ── Full pipeline integration ─────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_demo_runs(self):
        """The demo pipeline should run end-to-end without exceptions."""
        from pipeline import run_demo
        result = run_demo(CONFIG)
        assert "violations_found" in result
        assert "latency_ms" in result
        assert result["latency_ms"] > 0
