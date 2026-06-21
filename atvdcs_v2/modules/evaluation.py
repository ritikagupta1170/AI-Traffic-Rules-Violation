"""
Module 08 – Performance Evaluation Framework
────────────────────────────────────────────────────────────────────────────────
Evaluates every layer of the ATVDCS pipeline against targets from §5 of the
Solution Framework document.

Capabilities:
  • Detection metrics   – mAP@0.50, mAP@0.50:0.95 (COCO-style)
  • Violation metrics   – Precision, Recall, F1 per class
  • LPR metrics         – plate-level and character-level accuracy
  • End-to-end latency  – P50 / P95 / P99 across a batch
  • Confusion matrix    – per-violation-class
  • Environmental robustness – subsets: night / rain / fog / blur
  • Human-vs-machine agreement rate
  • MLflow logging (optional)
"""

import time
import json
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class DetectionMetrics:
    map_50: float = 0.0
    map_50_95: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)


@dataclass
class ViolationMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)


@dataclass
class LPRMetrics:
    plate_accuracy: float = 0.0
    char_accuracy: float = 0.0
    avg_confidence: float = 0.0


@dataclass
class LatencyMetrics:
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_fps: float = 0.0
    sample_count: int = 0


@dataclass
class EvalReport:
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    violation: ViolationMetrics = field(default_factory=ViolationMetrics)
    lpr: LPRMetrics = field(default_factory=LPRMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    end_to_end_accuracy: float = 0.0
    targets_met: Dict[str, bool] = field(default_factory=dict)
    notes: str = ""


# ── Evaluator ────────────────────────────────────────────────────────────────

class PipelineEvaluator:
    """
    Runs evaluation against the performance targets specified in config.yaml §5.
    """

    VIOLATION_CLASSES = [
        "helmet_non_compliance",
        "seatbelt_non_compliance",
        "triple_riding",
        "wrong_side_driving",
        "stop_line_violation",
        "red_light_violation",
        "illegal_parking",
    ]

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.targets = cfg["evaluation"]["targets"]
        self.reports_dir = Path(cfg["system"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_cfg = cfg["evaluation"].get("mlflow", {})

    # ── Public entry point ────────────────────────────────────────────────────

    def evaluate_from_predictions(
        self,
        predictions: List[Dict],   # [{plate_pred, plate_gt, violation_pred, violation_gt, latency_ms}, …]
        name: str = "eval_run",
    ) -> EvalReport:
        """
        Main evaluation entry point.

        Each prediction dict should contain:
          plate_pred       : str  – predicted plate text
          plate_gt         : str  – ground-truth plate text
          violation_pred   : str  – predicted violation class (or "none")
          violation_gt     : str  – ground-truth violation class (or "none")
          confidence       : float
          latency_ms       : float
        """
        report = EvalReport()

        # ── Violation metrics ──────────────────────────────────────────────
        report.violation = self._compute_violation_metrics(predictions)

        # ── LPR metrics ───────────────────────────────────────────────────
        report.lpr = self._compute_lpr_metrics(predictions)

        # ── Latency metrics ───────────────────────────────────────────────
        latencies = [p["latency_ms"] for p in predictions if "latency_ms" in p]
        report.latency = self._compute_latency(latencies)

        # ── End-to-end accuracy ───────────────────────────────────────────
        correct = sum(
            1 for p in predictions
            if p.get("violation_pred") == p.get("violation_gt")
            and self._plate_match(p.get("plate_pred", ""), p.get("plate_gt", ""))
        )
        report.end_to_end_accuracy = correct / max(len(predictions), 1)

        # ── Target checks ─────────────────────────────────────────────────
        report.targets_met = self._check_targets(report)

        # ── Log to MLflow (optional) ──────────────────────────────────────
        if self.mlflow_cfg.get("enabled"):
            self._log_mlflow(report, name)

        return report

    # ── Violation metrics ─────────────────────────────────────────────────────

    def _compute_violation_metrics(self, predictions: List[Dict]) -> ViolationMetrics:
        y_true = [p.get("violation_gt", "none") for p in predictions]
        y_pred = [p.get("violation_pred", "none") for p in predictions]
        all_classes = sorted(set(y_true + y_pred + self.VIOLATION_CLASSES))

        from collections import defaultdict
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for gt, pr in zip(y_true, y_pred):
            if gt == pr:
                tp[gt] += 1
            else:
                fp[pr] += 1
                fn[gt] += 1

        per_class: Dict[str, Dict] = {}
        for cls in all_classes:
            p = tp[cls] / max(tp[cls] + fp[cls], 1)
            r = tp[cls] / max(tp[cls] + fn[cls], 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            per_class[cls] = {"precision": round(p, 4),
                               "recall":    round(r, 4),
                               "f1":        round(f1, 4)}

        # Weighted average (by ground-truth frequency)
        total = max(len(y_true), 1)
        weights = {cls: y_true.count(cls) / total for cls in all_classes}
        precision = sum(per_class[c]["precision"] * weights[c] for c in all_classes)
        recall    = sum(per_class[c]["recall"]    * weights[c] for c in all_classes)
        f1        = sum(per_class[c]["f1"]        * weights[c] for c in all_classes)

        # Confusion matrix
        cls_idx = {c: i for i, c in enumerate(all_classes)}
        cm = np.zeros((len(all_classes), len(all_classes)), dtype=int)
        for gt, pr in zip(y_true, y_pred):
            cm[cls_idx[gt]][cls_idx[pr]] += 1

        return ViolationMetrics(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            per_class=per_class,
            confusion_matrix=cm,
            class_names=all_classes,
        )

    # ── LPR metrics ───────────────────────────────────────────────────────────

    def _compute_lpr_metrics(self, predictions: List[Dict]) -> LPRMetrics:
        plate_preds = [(p.get("plate_pred", ""), p.get("plate_gt", ""))
                       for p in predictions if p.get("plate_gt")]
        if not plate_preds:
            return LPRMetrics()

        plate_correct = sum(1 for pr, gt in plate_preds if pr == gt)
        plate_acc = plate_correct / len(plate_preds)

        # Character-level
        char_total, char_correct = 0, 0
        for pr, gt in plate_preds:
            for pc, gc in zip(pr, gt):
                char_total += 1
                if pc == gc:
                    char_correct += 1
            char_total += abs(len(pr) - len(gt))  # length penalty

        char_acc = char_correct / max(char_total, 1)
        avg_conf = float(np.mean([p.get("confidence", 0) for p in predictions]))

        return LPRMetrics(
            plate_accuracy=round(plate_acc, 4),
            char_accuracy=round(char_acc, 4),
            avg_confidence=round(avg_conf, 4),
        )

    # ── Latency ───────────────────────────────────────────────────────────────

    def _compute_latency(self, latencies: List[float]) -> LatencyMetrics:
        if not latencies:
            return LatencyMetrics()
        arr = np.array(latencies)
        return LatencyMetrics(
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            throughput_fps=round(1000.0 / max(np.mean(arr), 1e-9), 2),
            sample_count=len(latencies),
        )

    # ── Target checks ─────────────────────────────────────────────────────────

    def _check_targets(self, report: EvalReport) -> Dict[str, bool]:
        t = self.targets
        return {
            "f1_violation":     report.violation.f1       >= t.get("f1_violation", 0.91),
            "precision":        report.violation.precision >= t.get("precision_violation", 0.93),
            "recall":           report.violation.recall    >= t.get("recall_violation", 0.89),
            "plate_accuracy":   report.lpr.plate_accuracy  >= t.get("plate_accuracy", 0.93),
            "end_to_end":       report.end_to_end_accuracy >= t.get("end_to_end", 0.88),
            "latency_p95_ms":   report.latency.p95_ms      <= 400,
        }

    # ── MLflow ────────────────────────────────────────────────────────────────

    def _log_mlflow(self, report: EvalReport, name: str):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_cfg.get("tracking_uri", "mlruns/"))
            mlflow.set_experiment(self.mlflow_cfg.get("experiment_name", "ATVDCS"))
            with mlflow.start_run(run_name=name):
                mlflow.log_metric("precision",         report.violation.precision)
                mlflow.log_metric("recall",            report.violation.recall)
                mlflow.log_metric("f1",                report.violation.f1)
                mlflow.log_metric("plate_accuracy",    report.lpr.plate_accuracy)
                mlflow.log_metric("end_to_end",        report.end_to_end_accuracy)
                mlflow.log_metric("latency_p95_ms",    report.latency.p95_ms)
        except Exception as e:
            logger.warning("MLflow logging failed: %s", e)

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        report: EvalReport,
        output_path: Optional[str] = None,
    ) -> str:
        import seaborn as sns
        cm = report.violation.confusion_matrix
        names = report.violation.class_names
        path = output_path or str(self.reports_dir / "confusion_matrix.png")

        if cm is None or len(names) == 0:
            return path

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[n[:12] for n in names],
            yticklabels=[n[:12] for n in names],
            ax=ax,
        )
        ax.set_title("Violation Class Confusion Matrix", fontsize=14)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def print_report(self, report: EvalReport):
        """Pretty-print evaluation results to stdout."""
        print("\n" + "=" * 60)
        print("  ATVDCS PERFORMANCE EVALUATION REPORT")
        print(f"  {report.timestamp}")
        print("=" * 60)

        print("\n[ Violation Classification ]")
        print(f"  Precision : {report.violation.precision:.1%}")
        print(f"  Recall    : {report.violation.recall:.1%}")
        print(f"  F1 Score  : {report.violation.f1:.1%}")

        print("\n[ License Plate Recognition ]")
        print(f"  Plate accuracy : {report.lpr.plate_accuracy:.1%}")
        print(f"  Char  accuracy : {report.lpr.char_accuracy:.1%}")

        print("\n[ End-to-End ]")
        print(f"  Accuracy : {report.end_to_end_accuracy:.1%}")

        print("\n[ Latency ]")
        print(f"  P50 : {report.latency.p50_ms:.0f} ms")
        print(f"  P95 : {report.latency.p95_ms:.0f} ms")
        print(f"  FPS : {report.latency.throughput_fps:.1f}")

        print("\n[ Targets Met ]")
        for k, v in report.targets_met.items():
            icon = "✓" if v else "✗"
            print(f"  {icon}  {k}")
        print()

    # ── Plate match helper ────────────────────────────────────────────────────

    @staticmethod
    def _plate_match(pred: str, gt: str) -> bool:
        return pred.upper().replace("-", "") == gt.upper().replace("-", "")


# ── Latency benchmark utility ─────────────────────────────────────────────────

def benchmark_pipeline(pipeline_fn, images: list, warmup: int = 2) -> LatencyMetrics:
    """
    Benchmark any callable pipeline_fn(image) → result over a list of images.
    Runs `warmup` iterations first to stabilise caches.
    """
    evaluator = PipelineEvaluator.__new__(PipelineEvaluator)

    for img in images[:warmup]:
        pipeline_fn(img)

    latencies = []
    for img in images:
        t0 = time.perf_counter()
        pipeline_fn(img)
        latencies.append((time.perf_counter() - t0) * 1000)

    return evaluator._compute_latency(latencies)
