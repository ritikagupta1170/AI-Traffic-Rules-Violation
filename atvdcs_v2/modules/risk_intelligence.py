"""
Module 11 – Traffic Risk Intelligence
────────────────────────────────────────────────────────────────────────────────
Computes an Area Risk Score (0-100) per camera/junction by combining:
  • Violation volume        (normalized vs. city-wide max)
  • Violation severity       (uses the same severity weights as offender scoring)
  • Repeat-offender presence (share of violations caused by known repeat offenders)
  • Traffic density          (from cameras.traffic_density — sensor/manual feed)

This is the module that turns raw counts into a prioritization tool: "where
should the next patrol / camera upgrade / signage budget go?" — the kind of
decision-support framing that differentiates this from a plain detection demo.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import yaml

from modules.offender_profiling import DEFAULT_SEVERITY_WEIGHTS

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS area_risk_scores (
    camera_id              TEXT,
    location_label          TEXT,
    period_days             INTEGER,
    violation_count         INTEGER,
    severity_index          REAL,
    repeat_offender_ratio   REAL,
    traffic_density          REAL,
    area_risk_score          REAL,
    rank                     INTEGER,
    computed_at              TEXT,
    PRIMARY KEY (camera_id, period_days)
);
"""

DEFAULT_WEIGHTS = {
    "volume": 0.35,
    "severity": 0.25,
    "repeat_offenders": 0.20,
    "traffic_density": 0.20,
}


class AreaRiskScorer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        db_url = cfg["analytics"]["db_url"]
        self.db_path = db_url[len("sqlite:///"):] if db_url.startswith("sqlite:///") else db_url

        ri_cfg = cfg.get("risk_intelligence", {})
        self.weights = {**DEFAULT_WEIGHTS, **ri_cfg.get("component_weights", {})}
        self.repeat_threshold = ri_cfg.get("repeat_offender_min_violations", 2)

        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)

    def _load(self, days: int) -> pd.DataFrame:
        since = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
        with self._conn() as conn:
            df = pd.read_sql_query(
                """SELECT v.*, c.junction_name, c.traffic_density
                   FROM violations v LEFT JOIN cameras c ON v.camera_id = c.camera_id
                   WHERE v.timestamp >= ?""",
                conn, params=(since,),
            )
        if df.empty:
            return df
        df["location_label"] = df["junction_name"].fillna(df["camera_id"])
        df["traffic_density"] = df["traffic_density"].fillna(0.5)
        return df

    def compute(self, days: int = 30) -> List[Dict[str, Any]]:
        df = self._load(days)
        if df.empty:
            return []

        # Repeat-offender plates within the window
        repeat_plates = set(
            df.groupby("plate_text").size().loc[lambda s: s >= self.repeat_threshold].index
        )

        rows = []
        max_count = df.groupby("camera_id").size().max()

        for camera_id, group in df.groupby("camera_id"):
            count = len(group)
            severity = group["violation_class"].map(DEFAULT_SEVERITY_WEIGHTS).fillna(3.0).mean()
            severity_norm = min(severity / 10.0, 1.0)
            repeat_ratio = group["plate_text"].isin(repeat_plates).mean()
            density = float(group["traffic_density"].iloc[0])
            volume_norm = count / max_count if max_count else 0

            score = 100 * (
                self.weights["volume"] * volume_norm
                + self.weights["severity"] * severity_norm
                + self.weights["repeat_offenders"] * repeat_ratio
                + self.weights["traffic_density"] * density
            )

            rows.append({
                "camera_id": camera_id,
                "location_label": group["location_label"].iloc[0],
                "period_days": days,
                "violation_count": count,
                "severity_index": round(float(severity), 2),
                "repeat_offender_ratio": round(float(repeat_ratio), 3),
                "traffic_density": density,
                "area_risk_score": round(float(score), 1),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            })

        rows.sort(key=lambda r: r["area_risk_score"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i

        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO area_risk_scores
                   (camera_id, location_label, period_days, violation_count, severity_index,
                    repeat_offender_ratio, traffic_density, area_risk_score, rank, computed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(camera_id, period_days) DO UPDATE SET
                     location_label=excluded.location_label,
                     violation_count=excluded.violation_count,
                     severity_index=excluded.severity_index,
                     repeat_offender_ratio=excluded.repeat_offender_ratio,
                     traffic_density=excluded.traffic_density,
                     area_risk_score=excluded.area_risk_score,
                     rank=excluded.rank,
                     computed_at=excluded.computed_at
                """,
                [(r["camera_id"], r["location_label"], r["period_days"], r["violation_count"],
                  r["severity_index"], r["repeat_offender_ratio"], r["traffic_density"],
                  r["area_risk_score"], r["rank"], r["computed_at"]) for r in rows],
            )

        return rows

    def ranking(self, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM area_risk_scores WHERE period_days = ? ORDER BY rank LIMIT ?",
                conn, params=(days, limit),
            )
        return df.to_dict(orient="records")
