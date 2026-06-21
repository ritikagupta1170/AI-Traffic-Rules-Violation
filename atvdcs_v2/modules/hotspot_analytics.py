"""
Module 10 – Violation Hotspot Analytics
────────────────────────────────────────────────────────────────────────────────
Aggregates violations by camera / junction / location and time bucket to:
  • Rank hotspots (which junction/camera is worst, and for which violation type)
  • Produce heatmap-ready point data {lat, lon, weight} for the dashboard map
  • Surface hourly / daily / weekly / monthly trend series
  • Identify the peak violation period for each location

Requires a lightweight `cameras` lookup table (camera_id -> junction name,
location label, lat/lon) so hotspots are human-readable, not just camera IDs.
If a camera isn't registered, it still works — falls back to camera_id as
the location label and skips map plotting for that camera.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CAMERA_DDL = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id           TEXT PRIMARY KEY,
    junction_name        TEXT,
    location_name        TEXT,
    lat                  REAL,
    lon                  REAL,
    traffic_density       REAL DEFAULT 0.5   -- 0-1, from sensor feed or manual estimate
);
"""


class HotspotAnalyzer:
    """Location + time aggregation over the violations table."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        db_url = cfg["analytics"]["db_url"]
        self.db_path = db_url[len("sqlite:///"):] if db_url.startswith("sqlite:///") else db_url
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(CAMERA_DDL)

    # ── Camera / junction registry ───────────────────────────────────────────

    def upsert_camera(self, camera_id: str, junction_name: str, location_name: str,
                       lat: Optional[float] = None, lon: Optional[float] = None,
                       traffic_density: float = 0.5) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO cameras (camera_id, junction_name, location_name, lat, lon, traffic_density)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(camera_id) DO UPDATE SET
                     junction_name=excluded.junction_name,
                     location_name=excluded.location_name,
                     lat=excluded.lat, lon=excluded.lon,
                     traffic_density=excluded.traffic_density
                """,
                (camera_id, junction_name, location_name, lat, lon, traffic_density),
            )

    def _violations_with_location(self, days: int = 30) -> pd.DataFrame:
        since = (datetime.now(timezone.utc) - pd.Timedelta(days=days)).isoformat()
        with self._conn() as conn:
            df = pd.read_sql_query(
                """SELECT v.*, c.junction_name, c.location_name, c.lat AS cam_lat,
                          c.lon AS cam_lon, c.traffic_density
                   FROM violations v
                   LEFT JOIN cameras c ON v.camera_id = c.camera_id
                   WHERE v.timestamp >= ?""",
                conn, params=(since,),
            )
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["location_label"] = df["junction_name"].fillna(df["camera_id"])
        df["lat"] = df["lat"].fillna(df["cam_lat"])
        df["lon"] = df["lon"].fillna(df["cam_lon"])
        return df

    # ── Hotspot ranking ───────────────────────────────────────────────────────

    def hotspot_ranking(self, days: int = 30, limit: int = 20) -> List[Dict[str, Any]]:
        df = self._violations_with_location(days)
        if df.empty:
            return []

        grouped = df.groupby(["camera_id", "location_label"]).agg(
            violation_count=("record_id", "count"),
            avg_confidence=("violation_confidence", "mean"),
            top_violation_type=("violation_class", lambda s: s.value_counts().idxmax()),
        ).reset_index().sort_values("violation_count", ascending=False).head(limit)

        return grouped.to_dict(orient="records")

    # ── Heatmap data ──────────────────────────────────────────────────────────

    def heatmap_points(self, days: int = 30) -> List[Dict[str, Any]]:
        """Returns [{lat, lon, weight, camera_id, location_label}] for map overlays.
        Cameras without registered GPS are skipped (can't plot)."""
        df = self._violations_with_location(days)
        if df.empty:
            return []
        df = df.dropna(subset=["lat", "lon"])
        if df.empty:
            return []
        grouped = df.groupby(["camera_id", "location_label", "lat", "lon"]).size()
        grouped = grouped.reset_index(name="weight")
        return grouped.to_dict(orient="records")

    # ── Time-based trends ─────────────────────────────────────────────────────

    def trends(self, days: int = 30) -> Dict[str, Any]:
        df = self._violations_with_location(days)
        if df.empty:
            return {"hourly": {}, "daily_of_week": {}, "weekly": {}, "monthly": {}, "peak_hour": None}

        hourly = df.groupby(df["timestamp"].dt.hour).size().to_dict()
        daily_of_week = (
            df.groupby(df["timestamp"].dt.day_name()).size().to_dict()
        )
        weekly = df.groupby(df["timestamp"].dt.strftime("%G-W%V")).size().to_dict()
        monthly = df.groupby(df["timestamp"].dt.strftime("%Y-%m")).size().to_dict()

        peak_hour = int(max(hourly, key=hourly.get)) if hourly else None

        return {
            "hourly": hourly,
            "daily_of_week": daily_of_week,
            "weekly": weekly,
            "monthly": monthly,
            "peak_hour": peak_hour,
        }

    def peak_period_by_location(self, days: int = 30) -> List[Dict[str, Any]]:
        """For each location, find the hour-of-day with the most violations
        (drives the 'High probability of X near Junction Y between 5-7 PM' copy)."""
        df = self._violations_with_location(days)
        if df.empty:
            return []
        df["hour"] = df["timestamp"].dt.hour
        results = []
        for (loc, vtype), group in df.groupby(["location_label", "violation_class"]):
            hour_counts = group.groupby("hour").size()
            peak_hour = int(hour_counts.idxmax())
            results.append({
                "location": loc,
                "violation_type": vtype,
                "peak_hour_start": peak_hour,
                "peak_hour_end": (peak_hour + 2) % 24,
                "occurrences": int(hour_counts.max()),
                "total_for_type_here": int(group.shape[0]),
            })
        return sorted(results, key=lambda r: r["total_for_type_here"], reverse=True)
