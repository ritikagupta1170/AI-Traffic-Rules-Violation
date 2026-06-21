"""
Module 12 – Predictive Violation Analytics
────────────────────────────────────────────────────────────────────────────────
Hackathon-pragmatic forecasting: rather than standing up a full time-series
model (Prophet/ARIMA), we build an hour-of-week violation frequency table per
(camera, violation_type) from history. This is statistically defensible
(captures rush-hour / weekday-weekend seasonality), cheap to compute, fully
explainable to judges, and good enough to drive real alerts like:

    "High probability of helmet violations near Junction A between 5–7 PM."

Swap-in path to production: replace `_hour_of_week_frequency` with a proper
seasonal model (Facebook Prophet, or a Poisson GLM with hour/day-of-week/
camera as features) without touching the alerting/API layer.
"""

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id               TEXT,
    location_label          TEXT,
    violation_type          TEXT,
    predicted_window_start  TEXT,
    predicted_window_end    TEXT,
    probability             REAL,
    alert_text              TEXT,
    generated_at             TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_generated ON predictions(generated_at);
"""


class PredictiveAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        db_url = cfg["analytics"]["db_url"]
        self.db_path = db_url[len("sqlite:///"):] if db_url.startswith("sqlite:///") else db_url

        pred_cfg = cfg.get("predictive_analytics", {})
        self.history_days = pred_cfg.get("history_window_days", 90)
        self.alert_threshold = pred_cfg.get("alert_probability_threshold", 0.6)
        self.window_hours = pred_cfg.get("forecast_window_hours", 2)

        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)

    def _history(self) -> pd.DataFrame:
        since = (datetime.now(timezone.utc) - timedelta(days=self.history_days)).isoformat()
        with self._conn() as conn:
            df = pd.read_sql_query(
                """SELECT v.camera_id, v.violation_class, v.timestamp,
                          COALESCE(c.junction_name, v.camera_id) AS location_label
                   FROM violations v LEFT JOIN cameras c ON v.camera_id = c.camera_id
                   WHERE v.timestamp >= ?""",
                conn, params=(since,),
            )
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["dow"] = df["timestamp"].dt.dayofweek
        df["hour"] = df["timestamp"].dt.hour
        return df

    def _hour_of_week_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Count of violations per (camera, type, day-of-week, hour) bucket,
        normalized 0-1 within each (camera, type) group."""
        counts = (
            df.groupby(["camera_id", "location_label", "violation_class", "dow", "hour"])
            .size().reset_index(name="count")
        )
        counts["probability"] = counts.groupby(
            ["camera_id", "violation_class"]
        )["count"].transform(lambda s: s / s.max())
        return counts

    # ── Public API ────────────────────────────────────────────────────────────

    def forecast_next_hours(self, horizon_hours: int = 24) -> List[Dict[str, Any]]:
        """Score the next N hours (current real-world dow/hour onward) and
        return any (camera, type, hour-slot) whose historical probability
        exceeds the alert threshold."""
        df = self._history()
        if df.empty:
            return []

        freq = self._hour_of_week_frequency(df)
        now = datetime.now(timezone.utc)
        alerts = []

        for h in range(horizon_hours):
            slot_time = now + timedelta(hours=h)
            dow, hour = slot_time.weekday(), slot_time.hour
            matches = freq[(freq["dow"] == dow) & (freq["hour"] == hour) &
                           (freq["probability"] >= self.alert_threshold)]
            for _, row in matches.iterrows():
                alerts.append({
                    "camera_id": row["camera_id"],
                    "location_label": row["location_label"],
                    "violation_type": row["violation_class"],
                    "predicted_window_start": slot_time.replace(minute=0, second=0, microsecond=0).isoformat(),
                    "predicted_window_end": (slot_time + timedelta(hours=self.window_hours)).isoformat(),
                    "probability": round(float(row["probability"]), 2),
                    "alert_text": self._format_alert(row, slot_time),
                })

        # De-duplicate camera+type, keep highest probability
        dedup: Dict[str, Dict[str, Any]] = {}
        for a in alerts:
            key = f"{a['camera_id']}|{a['violation_type']}"
            if key not in dedup or a["probability"] > dedup[key]["probability"]:
                dedup[key] = a

        results = sorted(dedup.values(), key=lambda a: a["probability"], reverse=True)
        self._persist(results)
        return results

    def _format_alert(self, row, slot_time: datetime) -> str:
        vtype = row["violation_class"].replace("_", " ")
        start_h = slot_time.strftime("%I %p").lstrip("0")
        end_h = (slot_time + timedelta(hours=self.window_hours)).strftime("%I %p").lstrip("0")
        pct = int(row["probability"] * 100)
        return (f"High probability ({pct}%) of {vtype} near {row['location_label']} "
                f"between {start_h} and {end_h}.")

    def _persist(self, alerts: List[Dict[str, Any]]) -> None:
        if not alerts:
            return
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO predictions
                   (camera_id, location_label, violation_type, predicted_window_start,
                    predicted_window_end, probability, alert_text, generated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                [(a["camera_id"], a["location_label"], a["violation_type"],
                  a["predicted_window_start"], a["predicted_window_end"],
                  a["probability"], a["alert_text"], now_iso) for a in alerts],
            )

    def recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY generated_at DESC LIMIT ?",
                conn, params=(limit,),
            )
        return df.to_dict(orient="records")
