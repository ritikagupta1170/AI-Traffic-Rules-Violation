"""
Module 07 – Analytics & Reporting
────────────────────────────────────────────────────────────────────────────────
Stores violation records, provides query/export APIs, and generates:
  • Real-time summary statistics
  • Daily PDF report
  • CSV export
  • Matplotlib charts (violation trends, heatmaps, camera health)

Uses SQLite in dev/CPU mode; swap DB_URL in config for PostgreSQL + TimescaleDB
in production.
"""

import json
import yaml
import logging
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from modules.evidence import ViolationRecord

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS violations (
    record_id           TEXT PRIMARY KEY,
    vehicle_class       TEXT,
    plate_text          TEXT,
    plate_confidence    REAL,
    violation_class     TEXT,
    violation_confidence REAL,
    disposition         TEXT,
    lat                 REAL,
    lon                 REAL,
    timestamp           TEXT,
    camera_id           TEXT,
    pipeline_run_id     TEXT,
    model_version       TEXT,
    image_hash          TEXT,
    annotated_image_path TEXT,
    preview_image_path  TEXT,
    metadata            TEXT
);

CREATE INDEX IF NOT EXISTS idx_timestamp     ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_camera        ON violations(camera_id);
CREATE INDEX IF NOT EXISTS idx_violation_cls ON violations(violation_class);
CREATE INDEX IF NOT EXISTS idx_plate         ON violations(plate_text);
CREATE INDEX IF NOT EXISTS idx_disposition   ON violations(disposition);
"""


class AnalyticsDB:
    """SQLite-backed violation record store with pandas query helpers."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db_url: str = cfg["analytics"]["db_url"]
        # Accept sqlite:/// prefix or bare file path
        if db_url.startswith("sqlite:///"):
            db_path = db_url[len("sqlite:///"):]
        else:
            db_path = db_url

        self.db_path = db_path
        self.reports_dir = Path(cfg["system"]["reports_dir"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)
        logger.info("Analytics DB ready: %s", self.db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert(self, record: ViolationRecord) -> bool:
        """Insert a ViolationRecord. Returns True on success."""
        d = asdict(record)
        d["metadata"] = json.dumps(d["metadata"])
        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        sql = f"INSERT OR IGNORE INTO violations ({cols}) VALUES ({placeholders})"
        try:
            with self._conn() as conn:
                conn.execute(sql, list(d.values()))
            return True
        except Exception as e:
            logger.error("DB insert failed: %s", e)
            return False

    def insert_many(self, records: List[ViolationRecord]) -> int:
        return sum(1 for r in records if self.insert(r))

    # ── Read / Query ──────────────────────────────────────────────────────────

    def query(
        self,
        camera_id: Optional[str] = None,
        violation_class: Optional[str] = None,
        plate_text: Optional[str] = None,
        disposition: Optional[str] = None,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Flexible query returning a pandas DataFrame."""
        clauses, params = [], []
        if camera_id:
            clauses.append("camera_id = ?"); params.append(camera_id)
        if violation_class:
            clauses.append("violation_class = ?"); params.append(violation_class)
        if plate_text:
            clauses.append("plate_text LIKE ?"); params.append(f"%{plate_text}%")
        if disposition:
            clauses.append("disposition = ?"); params.append(disposition)
        if start_ts:
            clauses.append("timestamp >= ?"); params.append(start_ts)
        if end_ts:
            clauses.append("timestamp <= ?"); params.append(end_ts)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM violations {where} ORDER BY timestamp DESC LIMIT {limit}"
        with self._conn() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        return df

    def summary_stats(self, days: int = 7) -> Dict[str, Any]:
        """Return aggregated statistics for the last N days."""
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        df = self.query(start_ts=since, limit=10_000)
        if df.empty:
            return {"total": 0, "by_type": {}, "by_camera": {}, "by_disposition": {}}

        return {
            "total": len(df),
            "period_days": days,
            "by_type": df["violation_class"].value_counts().to_dict(),
            "by_camera": df["camera_id"].value_counts().to_dict(),
            "by_disposition": df["disposition"].value_counts().to_dict(),
            "avg_confidence": round(float(df["violation_confidence"].mean()), 3),
            "unique_plates": int(df["plate_text"].nunique()),
        }

    # ── Export ────────────────────────────────────────────────────────────────

    def export_csv(
        self,
        output_path: Optional[str] = None,
        **query_kwargs,
    ) -> str:
        df = self.query(**query_kwargs)
        path = output_path or str(self.reports_dir / f"export_{_ts()}.csv")
        df.to_csv(path, index=False)
        logger.info("CSV exported: %s (%d rows)", path, len(df))
        return path

    # ── Charts ────────────────────────────────────────────────────────────────

    def plot_daily_trend(
        self, days: int = 30, output_path: Optional[str] = None
    ) -> str:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        df = self.query(start_ts=since, limit=50_000)
        path = output_path or str(self.reports_dir / f"trend_{_ts()}.png")

        if df.empty:
            _save_placeholder(path, "No data for trend chart")
            return path

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        daily = df.groupby(["date", "violation_class"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        daily.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_title(f"Daily Violations — Last {days} Days", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_violation_distribution(
        self, days: int = 7, output_path: Optional[str] = None
    ) -> str:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        df = self.query(start_ts=since, limit=50_000)
        path = output_path or str(self.reports_dir / f"dist_{_ts()}.png")

        if df.empty:
            _save_placeholder(path, "No data for distribution chart")
            return path

        counts = df["violation_class"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        counts.plot(kind="barh", ax=ax, color=sns.color_palette("husl", len(counts)))
        ax.set_title(f"Violation Distribution — Last {days} Days", fontsize=13)
        ax.set_xlabel("Count")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    def plot_confidence_histogram(
        self, output_path: Optional[str] = None
    ) -> str:
        df = self.query(limit=5_000)
        path = output_path or str(self.reports_dir / f"conf_hist_{_ts()}.png")

        if df.empty:
            _save_placeholder(path, "No data for confidence histogram")
            return path

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["violation_confidence"], bins=20, color="#2196F3", edgecolor="white")
        ax.axvline(0.90, color="green",  linestyle="--", label="Auto-enforce (0.90)")
        ax.axvline(0.70, color="orange", linestyle="--", label="Human review (0.70)")
        ax.set_title("Violation Confidence Distribution", fontsize=13)
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path

    # ── PDF Report ────────────────────────────────────────────────────────────

    def generate_pdf_report(
        self, days: int = 1, output_path: Optional[str] = None
    ) -> str:
        """Generate a daily summary PDF report."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            )
            from reportlab.lib.styles import getSampleStyleSheet
        except ImportError:
            logger.warning("reportlab not installed — skipping PDF report")
            return ""

        stats = self.summary_stats(days=days)
        path = output_path or str(self.reports_dir / f"report_{_ts()}.pdf")
        doc = SimpleDocTemplate(path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("ATVDCS – Daily Summary Report", styles["Title"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"Period: last {days} day(s)", styles["Normal"]
        ))
        story.append(Spacer(1, 0.5 * cm))

        # Summary table
        table_data = [["Metric", "Value"]]
        table_data += [
            ["Total Violations", str(stats["total"])],
            ["Avg Confidence",   str(stats.get("avg_confidence", "—"))],
            ["Unique Plates",    str(stats.get("unique_plates", "—"))],
        ]
        for k, v in stats.get("by_type", {}).items():
            table_data.append([f"  {k.replace('_', ' ').title()}", str(v)])
        for k, v in stats.get("by_disposition", {}).items():
            table_data.append([f"  Disposition: {k}", str(v)])

        tbl = Table(table_data, colWidths=[10 * cm, 6 * cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#E3F2FD")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tbl)
        doc.build(story)
        logger.info("PDF report saved: %s", path)
        return path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_placeholder(path: str, message: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(path, dpi=100)
    plt.close()
