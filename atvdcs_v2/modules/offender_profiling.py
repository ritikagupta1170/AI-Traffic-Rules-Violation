"""
Module 09 – Repeat Offender Detection & Vehicle Risk Scoring
────────────────────────────────────────────────────────────────────────────────
Builds a behavioural profile for every license plate seen by the system and
computes a 0–100 Vehicle Risk Score that powers the "Top Repeat Offenders"
dashboard and feeds Area Risk Intelligence (module 11).

Design notes
------------
* Reads directly from the existing `violations` table (no schema break).
* Writes a derived, cheap-to-query `offender_risk_scores` table so the
  dashboard can sort/filter thousands of plates without recomputing on
  every request. Call `recompute_all()` on a schedule (e.g. every 5 min
  via APScheduler/cron) or after each new evidence insert.
* Severity weights & half-life are config-driven so judges can show the
  scoring logic is tunable, not a black box.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DDL = """
CREATE TABLE IF NOT EXISTS offender_risk_scores (
    plate_text              TEXT PRIMARY KEY,
    total_violations        INTEGER,
    distinct_violation_types INTEGER,
    first_seen              TEXT,
    last_violation_ts       TEXT,
    days_since_last         REAL,
    risk_score               REAL,
    risk_tier                TEXT,
    violation_breakdown      TEXT,   -- JSON {type: count}
    updated_at                TEXT
);
CREATE INDEX IF NOT EXISTS idx_offender_score ON offender_risk_scores(risk_score);
"""

# Severity weight per violation class (1 = minor, 10 = severe).
# Tunable from config.yaml -> offender_profiling.severity_weights
DEFAULT_SEVERITY_WEIGHTS: Dict[str, float] = {
    "red_light_violation": 10,
    "wrong_side_driving": 9,
    "triple_riding": 7,
    "stop_line_violation": 6,
    "helmet_non_compliance": 5,
    "seatbelt_non_compliance": 4,
    "illegal_parking": 3,
}

RISK_TIERS = [
    (85, "CRITICAL"),
    (60, "HIGH"),
    (30, "MEDIUM"),
    (0, "LOW"),
]


@dataclass
class OffenderProfile:
    plate_text: str
    total_violations: int
    distinct_violation_types: int
    first_seen: Optional[str]
    last_violation_ts: Optional[str]
    days_since_last: float
    risk_score: float
    risk_tier: str
    violation_breakdown: Dict[str, int] = field(default_factory=dict)


class OffenderProfiler:
    """Computes and persists repeat-offender risk profiles."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        db_url = cfg["analytics"]["db_url"]
        self.db_path = db_url[len("sqlite:///"):] if db_url.startswith("sqlite:///") else db_url

        op_cfg = cfg.get("offender_profiling", {})
        self.severity_weights = {**DEFAULT_SEVERITY_WEIGHTS, **op_cfg.get("severity_weights", {})}
        self.half_life_days = op_cfg.get("half_life_days", 90)   # recency decay
        self.saturation_k = op_cfg.get("saturation_k", 30)        # curve steepness
        self.variety_bonus_per_type = op_cfg.get("variety_bonus_per_type", 4)
        self.repeat_threshold = op_cfg.get("repeat_offender_min_violations", 2)

        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(DDL)

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _recency_factor(self, days_ago: float) -> float:
        """Exponential decay; recent violations weigh more than old ones."""
        factor = float(np.exp(-days_ago / max(self.half_life_days, 1)))
        return max(0.15, factor)   # floor so very old offences still count a little

    def _score_plate(self, plate_text: str, df: pd.DataFrame) -> OffenderProfile:
        now = datetime.now(timezone.utc)
        ts = pd.to_datetime(df["timestamp"], utc=True)
        days_ago = (now - ts).dt.total_seconds() / 86400.0

        weights = df["violation_class"].map(self.severity_weights).fillna(3.0)
        weighted = weights * days_ago.apply(self._recency_factor)
        weight_sum = float(weighted.sum())

        distinct_types = df["violation_class"].nunique()
        variety_bonus = self.variety_bonus_per_type * max(distinct_types - 1, 0)

        # Saturating curve: many small violations don't blow past 100,
        # but severe + frequent + varied offenders climb fast.
        base_score = 100 * (1 - np.exp(-weight_sum / self.saturation_k))
        risk_score = float(min(100.0, base_score + variety_bonus))

        tier = next(name for floor, name in RISK_TIERS if risk_score >= floor)

        return OffenderProfile(
            plate_text=plate_text,
            total_violations=len(df),
            distinct_violation_types=int(distinct_types),
            first_seen=str(ts.min()),
            last_violation_ts=str(ts.max()),
            days_since_last=round(float(days_ago.min()), 2),
            risk_score=round(risk_score, 1),
            risk_tier=tier,
            violation_breakdown=df["violation_class"].value_counts().to_dict(),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def recompute_all(self) -> int:
        """Recompute risk scores for every plate seen in `violations`. Returns count."""
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT plate_text, violation_class, timestamp FROM violations "
                "WHERE plate_text IS NOT NULL AND plate_text != 'UNREADABLE'",
                conn,
            )
        if df.empty:
            return 0

        rows = []
        for plate, group in df.groupby("plate_text"):
            profile = self._score_plate(plate, group)
            rows.append((
                profile.plate_text, profile.total_violations, profile.distinct_violation_types,
                profile.first_seen, profile.last_violation_ts, profile.days_since_last,
                profile.risk_score, profile.risk_tier,
                json.dumps(profile.violation_breakdown),
                datetime.now(timezone.utc).isoformat(),
            ))

        with self._conn() as conn:
            conn.executemany(
                """INSERT INTO offender_risk_scores
                   (plate_text, total_violations, distinct_violation_types, first_seen,
                    last_violation_ts, days_since_last, risk_score, risk_tier,
                    violation_breakdown, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(plate_text) DO UPDATE SET
                     total_violations=excluded.total_violations,
                     distinct_violation_types=excluded.distinct_violation_types,
                     first_seen=excluded.first_seen,
                     last_violation_ts=excluded.last_violation_ts,
                     days_since_last=excluded.days_since_last,
                     risk_score=excluded.risk_score,
                     risk_tier=excluded.risk_tier,
                     violation_breakdown=excluded.violation_breakdown,
                     updated_at=excluded.updated_at
                """,
                rows,
            )
        logger.info("Recomputed risk scores for %d plates", len(rows))
        return len(rows)

    def top_offenders(self, limit: int = 20, min_violations: Optional[int] = None) -> pd.DataFrame:
        min_v = min_violations or self.repeat_threshold
        with self._conn() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM offender_risk_scores WHERE total_violations >= ? "
                "ORDER BY risk_score DESC LIMIT ?",
                conn, params=(min_v, limit),
            )
        return df

    def get_profile(self, plate_text: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM offender_risk_scores WHERE plate_text = ?", (plate_text,)
            )
            row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["violation_breakdown"] = json.loads(d["violation_breakdown"] or "{}")
        return d

    def repeat_offender_count(self) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM offender_risk_scores WHERE total_violations >= ?",
                (self.repeat_threshold,),
            )
            return cur.fetchone()[0]
