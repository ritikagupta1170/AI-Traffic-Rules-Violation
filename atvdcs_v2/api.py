"""
ATVDCS – REST API Server
────────────────────────────────────────────────────────────────────────────────
FastAPI endpoints for integration with external enforcement systems (MoRTH,
State Transport Authority, court management systems).

Start:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /process          – upload image, get violation analysis
  GET  /violations       – query violation records with filters
  GET  /stats            – summary statistics
  GET  /report/pdf       – trigger daily PDF report generation
  GET  /health           – liveness check
"""

from modules import offender_profiling
from modules import offender_profiling
import io
import uuid
import logging
import importlib
from typing import Optional, List
from datetime import datetime
from fastapi.encoders import jsonable_encoder
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from pipeline import ATVDCSPipeline
from modules.explainability import explain_record
from modules.evidence import EvidenceGenerator, ViolationRecord

logger = logging.getLogger("atvdcs.api")

# Single shared pipeline instance (modules loaded once)
pipeline: Optional[ATVDCSPipeline] = None
offender_profiler = None
hotspot_analyzer = None
area_risk_scorer = None
predictive_analyzer = None


def _get_offender_profiler():
    global offender_profiler
    if offender_profiler is None:
        from modules.offender_profiling import OffenderProfiler
        offender_profiler = OffenderProfiler()
    return offender_profiler


def _get_hotspot_analyzer():
    global hotspot_analyzer
    if hotspot_analyzer is None:
        from modules.hotspot_analytics import HotspotAnalyzer
        hotspot_analyzer = HotspotAnalyzer()
    return hotspot_analyzer


def _get_area_risk_scorer():
    global area_risk_scorer
    if area_risk_scorer is None:
        from modules.risk_intelligence import AreaRiskScorer
        area_risk_scorer = AreaRiskScorer()
    return area_risk_scorer


def _get_predictive_analyzer():
    global predictive_analyzer
    if predictive_analyzer is None:
        from modules.predictive_analytics import PredictiveAnalyzer
        predictive_analyzer = PredictiveAnalyzer()
    return predictive_analyzer


app = FastAPI(
    title="ATVDCS API",
    description="AI Traffic Intelligence Platform (Automated Traffic Violation "
                "Detection, Classification & Risk Intelligence)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    global pipeline
    logger.info("Loading ATVDCS pipeline …")
    pipeline = ATVDCSPipeline()
    logger.info("API ready")


# ── Response models ───────────────────────────────────────────────────────────

class ViolationSummary(BaseModel):
    run_id: str
    camera_id: str
    violations_found: int
    auto_enforce: int
    human_review: int
    plate: Optional[str]
    plate_valid: bool
    records_saved: int
    latency_ms: float
    violation_confidence: Optional[float] = None


class StatsResponse(BaseModel):
    total: int
    period_days: int
    by_type: dict
    by_camera: dict
    by_disposition: dict
    avg_confidence: Optional[float]
    unique_plates: Optional[int]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/process", response_model=ViolationSummary)
async def process_image(
    file: UploadFile = File(...),
    camera_id: str = Query("CAM_001"),
    signal_red: bool = Query(False),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    """
    Upload a traffic image and receive violation analysis.

    - **file**: JPEG or PNG image
    - **camera_id**: identifier of the originating camera
    - **signal_red**: set True if traffic signal is currently RED
    - **lat / lon**: GPS coordinates of the camera
    """
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file")

    # Decode image
    arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "Could not decode image")

    gps = (lat, lon) if lat is not None and lon is not None else None
    result = pipeline.process_image(
        img,
        camera_id=camera_id,
        gps=gps,
        signal_red=signal_red,
    )

    return ViolationSummary(
        run_id=result["run_id"],
        camera_id=result["camera_id"],
        violations_found=result["violations_found"],
        auto_enforce=result["auto_enforce"],
        human_review=result["human_review"],
        plate=result["plate"],
        plate_valid=result["plate_valid"],
        records_saved=result["records_saved"],
        latency_ms=result["latency_ms"],
        violation_confidence=result["records"][0].violation_confidence if result["records"] else None,
    )


@app.get("/violations")
def get_violations(
    camera_id: Optional[str] = None,
    violation_class: Optional[str] = None,
    plate_text: Optional[str] = None,
    disposition: Optional[str] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: int = 100,
):
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    try:
        import pandas as pd
        import numpy as np
        import json

        df = pipeline.analytics_db.query(
            camera_id=camera_id,
            violation_class=violation_class,
            plate_text=plate_text,
            disposition=disposition,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
        )

        # Convert NaN/Inf everywhere
        df = df.astype(object)

        df = df.replace({
            np.nan: None,
            np.inf: None,
            -np.inf: None
        })

        records = df.to_dict(orient="records")

        # Clean nested structures recursively
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}

            if isinstance(obj, list):
                return [clean(v) for v in obj]

            if isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None

            return obj

        records = clean(records)

        # Validate JSON before returning
        json.dumps(records, allow_nan=False)

        return records

    except Exception as e:
        logger.exception("Failed to fetch violations")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/stats", response_model=StatsResponse)
def get_stats(days: int = Query(7, ge=1, le=365)):
    """Aggregated statistics for the last N days."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    stats = pipeline.analytics_db.summary_stats(days=days)
    return StatsResponse(**{
        "total": stats.get("total", 0),
        "period_days": days,
        "by_type": stats.get("by_type", {}),
        "by_camera": stats.get("by_camera", {}),
        "by_disposition": stats.get("by_disposition", {}),
        "avg_confidence": stats.get("avg_confidence"),
        "unique_plates": stats.get("unique_plates"),
    })


@app.get("/report/pdf")
def generate_report(days: int = Query(1, ge=1, le=30)):
    """Trigger PDF report generation and return the file."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    path = pipeline.analytics_db.generate_pdf_report(days=days)
    if not path:
        raise HTTPException(500, "PDF generation failed (reportlab not installed?)")
    return FileResponse(path, media_type="application/pdf",
                        filename=f"atvdcs_report_{days}d.pdf")


@app.get("/export/csv")
def export_csv(
    camera_id:       Optional[str] = None,
    violation_class: Optional[str] = None,
    limit:           int = 1000,
):
    """Export violation records as CSV."""
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    path = pipeline.analytics_db.export_csv(
        camera_id=camera_id, violation_class=violation_class, limit=limit
    )
    return FileResponse(path, media_type="text/csv", filename="atvdcs_export.csv")


# ── Repeat Offender Detection (Module 09) ────────────────────────────────────

@app.post("/offenders/recompute")
def recompute_offenders():
    """Recompute risk scores for every plate. Call after a batch of new
    evidence, or on a schedule (cron/APScheduler) in production."""
    n = _get_offender_profiler().recompute_all()
    return {"plates_scored": n}


@app.get("/offenders/top")
def top_offenders(limit: int = Query(20, ge=1, le=200),
                   min_violations: Optional[int] = None):
    """Top Repeat Offenders leaderboard, sorted by Vehicle Risk Score (0-100)."""
    df = _get_offender_profiler().top_offenders(limit=limit, min_violations=min_violations)
    return JSONResponse(content=df.to_dict(orient="records"))


@app.get("/offenders/{plate_text}")
def offender_profile(plate_text: str):
    """Full offender profile for a single plate: history, breakdown, risk tier."""
    profile = _get_offender_profiler().get_profile(plate_text.upper())
    if not profile:
        raise HTTPException(404, "No profile found for this plate")
    return JSONResponse(content=profile)


# ── Violation Hotspot Analytics (Module 10) ──────────────────────────────────

@app.post("/cameras")
def register_camera(camera_id: str, junction_name: str, location_name: str,
                     lat: Optional[float] = None, lon: Optional[float] = None,
                     traffic_density: float = 0.5):
    """Register/update a camera's junction name, location and GPS for
    human-readable hotspot labels and heatmap plotting."""
    _get_hotspot_analyzer().upsert_camera(camera_id, junction_name, location_name, lat, lon, traffic_density)
    return {"status": "ok", "camera_id": camera_id}


@app.get("/hotspots/ranking")
def hotspot_ranking(days: int = Query(30, ge=1, le=365), limit: int = Query(20, ge=1, le=200)):
    """Ranked list of worst locations by violation count."""
    return JSONResponse(content=_get_hotspot_analyzer().hotspot_ranking(days=days, limit=limit))


@app.get("/hotspots/heatmap")
def hotspot_heatmap(days: int = Query(30, ge=1, le=365)):
    """Heatmap-ready points: [{lat, lon, weight, camera_id, location_label}]."""
    return JSONResponse(content=_get_hotspot_analyzer().heatmap_points(days=days))


@app.get("/hotspots/trends")
def hotspot_trends(days: int = Query(30, ge=1, le=365)):
    """Hourly / day-of-week / weekly / monthly violation trend series + peak hour."""
    return JSONResponse(content=_get_hotspot_analyzer().trends(days=days))


@app.get("/hotspots/peak-periods")
def hotspot_peak_periods(days: int = Query(30, ge=1, le=365)):
    """Peak violation window per (location, violation type) — drives alert copy."""
    return JSONResponse(content=_get_hotspot_analyzer().peak_period_by_location(days=days))


# ── Traffic Risk Intelligence (Module 11) ────────────────────────────────────

@app.post("/risk/areas/recompute")
def recompute_area_risk(days: int = Query(30, ge=1, le=365)):
    """Recompute Area Risk Scores (volume + severity + repeat offenders + density)."""
    rows = _get_area_risk_scorer().compute(days=days)
    return {"locations_scored": len(rows)}


@app.get("/risk/areas")
def area_risk_ranking(days: int = Query(30, ge=1, le=365), limit: int = Query(20, ge=1, le=200)):
    """Ranked locations by Area Risk Score (0-100)."""
    return JSONResponse(content=_get_area_risk_scorer().ranking(days=days, limit=limit))


# ── Predictive Violation Analytics (Module 12) ───────────────────────────────

@app.get("/predictions/alerts")
def predictive_alerts(horizon_hours: int = Query(24, ge=1, le=168)):
    """Forecast the next N hours and return alerts for high-probability
    (location, violation type, time-window) combinations."""
    return JSONResponse(content=_get_predictive_analyzer().forecast_next_hours(horizon_hours=horizon_hours))


@app.get("/predictions/recent")
def recent_predictions(limit: int = Query(20, ge=1, le=200)):
    """Most recently generated predictive alerts (persisted history)."""
    return JSONResponse(content=_get_predictive_analyzer().recent_alerts(limit=limit))


# ── Explainable AI (Module 13) ───────────────────────────────────────────────

@app.get("/violations/{record_id}/explain")
def explain_violation_record(record_id: str):
    """Human-readable explanation: confidence, reason, supporting detections,
    disposition rationale — for a single violation record."""
    import json as _json
    row_df = pipeline.analytics_db.query(limit=10_000)
    match = row_df[row_df["record_id"] == record_id]
    if match.empty:
        raise HTTPException(404, "Record not found")
    metadata = _json.loads(match.iloc[0]["metadata"])
    return JSONResponse(content=explain_record(metadata))


# ── Enhanced Evidence Management (Module 06+) ────────────────────────────────

@app.get("/evidence/{record_id}/verify")
def verify_evidence(record_id: str):
    """Re-hash the stored annotated image and compare to the hash captured at
    generation time — tamper / integrity check for court-admissible evidence."""
    import json as _json
    df = pipeline.analytics_db.query(limit=10_000)
    match = df[df["record_id"] == record_id]
    if match.empty:
        raise HTTPException(404, "Record not found")
    row = match.iloc[0]
    record = ViolationRecord(
        record_id=row["record_id"], vehicle_class=row["vehicle_class"],
        plate_text=row["plate_text"], plate_confidence=row["plate_confidence"],
        violation_class=row["violation_class"], violation_confidence=row["violation_confidence"],
        disposition=row["disposition"], lat=row["lat"], lon=row["lon"],
        timestamp=row["timestamp"], camera_id=row["camera_id"],
        pipeline_run_id=row["pipeline_run_id"], model_version=row["model_version"],
        image_hash=row["image_hash"], annotated_image_path=row["annotated_image_path"],
        preview_image_path=row["preview_image_path"], metadata=_json.loads(row["metadata"]),
    )
    return JSONResponse(content=EvidenceGenerator.verify_integrity(record))

@app.get("/autodeploy-test")
def autodeploy_test():
    return {"status": "Auto deploy works"}