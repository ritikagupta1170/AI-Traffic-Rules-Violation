# ATVDCS v2 — From Violation Detector to AI Traffic Intelligence Platform

**Upgrade plan for your existing 9-module prototype (Preprocessing → Detection →
Violation Classification → LPR → Evidence → Analytics → FastAPI → Dashboard).**
Nothing below replaces your pipeline — every new module reads from the same
`violations` table your `AnalyticsDB` already writes to, and every new file is
additive. Four new backend modules are included as **working code** in this
delivery (see §6); the rest of this document is the architecture, schema, and
narrative you need to finish the job and pitch it.

---

## 1. What actually changes

| Layer | Before | After |
|---|---|---|
| Per-vehicle history | None — each violation is an isolated row | `offender_profiling.py` → risk score, tier, breakdown per plate |
| Spatial analysis | `camera_id` string only | `cameras` lookup table → junction name, GPS, density; hotspot ranking + heatmap |
| Time analysis | One daily trend chart | Hourly / day-of-week / weekly / monthly series + peak-period detection |
| Forecasting | None | Hour-of-week frequency model → "high probability of X near Y between 5–7 PM" alerts |
| Area prioritization | None | Composite Area Risk Score (volume + severity + repeat offenders + density) |
| Explainability | Confidence number only | Structured reason + supporting detections + disposition rationale per violation |
| Evidence trust | SHA-256 stored at write time, never re-checked | `/evidence/{id}/verify` re-hashes and confirms/denies integrity on demand |
| Dashboard | Single page | 7 dedicated pages (see §5) |
| API surface | 6 endpoints | 19 endpoints |

---

## 2. Updated architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              CAPTURE LAYER                                   │
│   IP/CCTV cameras  →  RTSP/frame-grab  →  Redis Stream (frame queue)         │
└───────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CORE DETECTION PIPELINE  (existing, unchanged)      │
│  01 Preprocessing → 02 Detection (YOLOv11/RT-DETR + ByteTrack)               │
│       → 03/04 Violation Classification → 05 LPR (PaddleOCR)                 │
│       → 06 Evidence Generation  ──────────┐                                 │
│                                            │ writes ViolationRecord          │
└────────────────────────────────────────────┼────────────────────────────────┘
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    07 ANALYTICS DB  (PostgreSQL in prod, SQLite in proto)    │
│                    table: violations  (+ cameras lookup table)              │
└───────┬─────────────┬─────────────┬─────────────┬─────────────┬────────────┘
        ▼             ▼             ▼             ▼             ▼
   ┌────────┐   ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐
   │ 09     │   │ 10       │  │ 11       │  │ 12        │  │ 13           │
   │Offender│   │Hotspot   │  │Area Risk │  │Predictive │  │Explainability│
   │Profiler│   │Analytics │  │Intel.    │  │Analytics  │  │Layer         │
   └───┬────┘   └────┬─────┘  └────┬─────┘  └─────┬─────┘  └──────┬───────┘
       │             │             │              │               │
       └─────────────┴──────┬──────┴──────────────┴───────────────┘
                             ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │  ← 19 endpoints (§4)
                    │  + Redis cache      │
                    └──────────┬──────────┘
                                ▼
                    ┌─────────────────────┐
                    │   React Dashboard   │  ← 7 pages (§5)
                    └─────────────────────┘
```

**New scheduled jobs** (cron / APScheduler / Celery beat — pick whichever is
fastest to wire for the demo):
- every 5 min → `OffenderProfiler.recompute_all()`
- every 15 min → `AreaRiskScorer.compute()`
- every hour → `PredictiveAnalyzer.forecast_next_hours()`

These are cheap (pandas group-bys over a few thousand rows) — no GPU, no
external service, safe to run on the same box as the API for a hackathon demo.

---

## 3. Database schema (full)

Existing table (unchanged):

```sql
CREATE TABLE violations (
    record_id            TEXT PRIMARY KEY,
    vehicle_class        TEXT,
    plate_text           TEXT,
    plate_confidence     REAL,
    violation_class      TEXT,
    violation_confidence REAL,
    disposition          TEXT,
    lat                  REAL,
    lon                  REAL,
    timestamp            TEXT,
    camera_id            TEXT,
    pipeline_run_id      TEXT,
    model_version        TEXT,
    image_hash           TEXT,
    annotated_image_path TEXT,
    preview_image_path   TEXT,
    metadata             TEXT        -- now also carries {"explanation": {...}}
);
```

New tables (created automatically by the new modules on first run):

```sql
-- Module 10: camera → junction/location/GPS lookup
CREATE TABLE cameras (
    camera_id        TEXT PRIMARY KEY,
    junction_name    TEXT,
    location_name    TEXT,
    lat              REAL,
    lon              REAL,
    traffic_density  REAL DEFAULT 0.5     -- 0-1, sensor feed or manual estimate
);

-- Module 09: derived per-plate risk profile (recomputed on schedule)
CREATE TABLE offender_risk_scores (
    plate_text               TEXT PRIMARY KEY,
    total_violations         INTEGER,
    distinct_violation_types INTEGER,
    first_seen               TEXT,
    last_violation_ts        TEXT,
    days_since_last          REAL,
    risk_score                REAL,        -- 0-100
    risk_tier                 TEXT,         -- LOW / MEDIUM / HIGH / CRITICAL
    violation_breakdown       TEXT,         -- JSON {type: count}
    updated_at                 TEXT
);

-- Module 11: derived per-location risk profile (recomputed on schedule)
CREATE TABLE area_risk_scores (
    camera_id             TEXT,
    location_label        TEXT,
    period_days           INTEGER,
    violation_count       INTEGER,
    severity_index        REAL,
    repeat_offender_ratio REAL,
    traffic_density       REAL,
    area_risk_score       REAL,            -- 0-100
    rank                  INTEGER,
    computed_at           TEXT,
    PRIMARY KEY (camera_id, period_days)
);

-- Module 12: persisted forecast/alert history
CREATE TABLE predictions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id               TEXT,
    location_label          TEXT,
    violation_type          TEXT,
    predicted_window_start  TEXT,
    predicted_window_end    TEXT,
    probability             REAL,
    alert_text              TEXT,
    generated_at            TEXT
);
```

Production swap: SQLite → **PostgreSQL + TimescaleDB** (the `violations` and
`predictions` tables are natural hypertables on `timestamp`). Add a Redis
layer in front of `offender_risk_scores` / `area_risk_scores` for sub-50ms
dashboard reads — these are read-heavy, write-light tables.

---

## 4. API endpoints (complete)

| Method | Path | Module | Description |
|---|---|---|---|
| POST | `/process` | core | Upload image → run full pipeline |
| GET | `/violations` | core | Query violation records |
| GET | `/stats` | core | Summary statistics |
| GET | `/report/pdf` | core | Daily PDF report |
| GET | `/export/csv` | core | CSV export |
| GET | `/health` | core | Liveness check |
| POST | `/offenders/recompute` | 09 | Recompute all risk scores |
| GET | `/offenders/top` | 09 | Top Repeat Offenders leaderboard |
| GET | `/offenders/{plate_text}` | 09 | Full offender profile |
| POST | `/cameras` | 10 | Register/update camera → junction/GPS |
| GET | `/hotspots/ranking` | 10 | Worst locations by violation count |
| GET | `/hotspots/heatmap` | 10 | Heatmap-ready `{lat, lon, weight}` points |
| GET | `/hotspots/trends` | 10 | Hourly/daily/weekly/monthly series + peak hour |
| GET | `/hotspots/peak-periods` | 10 | Peak window per (location, violation type) |
| POST | `/risk/areas/recompute` | 11 | Recompute Area Risk Scores |
| GET | `/risk/areas` | 11 | Ranked locations by Area Risk Score |
| GET | `/predictions/alerts` | 12 | Forecast next N hours, return high-probability alerts |
| GET | `/predictions/recent` | 12 | Recently generated alert history |
| GET | `/violations/{id}/explain` | 13 | Reason + supporting detections + disposition rationale |
| GET | `/evidence/{id}/verify` | 06+ | Re-hash and confirm/deny evidence integrity |

All 13 new endpoints are implemented in the updated `api.py` included in this
delivery — they're not a plan, they're callable today against your existing
pipeline once `OffenderProfiler`, `HotspotAnalyzer`, `AreaRiskScorer`, and
`PredictiveAnalyzer` are instantiated at startup (already wired in).

---

## 5. Dashboard design — 7 pages

1. **Live Monitoring** — current camera feed thumbnails, last-N processed
   frames, real-time violation ticker (poll `/violations?limit=20` every
   3–5s or upgrade to a WebSocket push from the pipeline).
2. **Violations Feed** — paginated table from `/violations`, filterable by
   camera/type/disposition/date; row-click opens the evidence package +
   `/violations/{id}/explain` panel (confidence, reason, supporting boxes).
3. **Repeat Offenders** — leaderboard from `/offenders/top`, risk-tier badge
   colors (LOW grey → CRITICAL red), click-through to `/offenders/{plate}`
   showing full violation history timeline.
4. **Hotspot Analysis** — map view fed by `/hotspots/heatmap` (Leaflet/Mapbox
   circle-weight markers), ranking table from `/hotspots/ranking`, trend
   charts from `/hotspots/trends` (hour-of-day bar, day-of-week bar).
5. **Risk Intelligence** — ranked location table from `/risk/areas` with a
   stacked bar showing the 4 score components (volume/severity/repeat/density)
   per location, so judges can see the score isn't a black box.
6. **Predictive Analytics** — alert cards from `/predictions/alerts`
   ("High probability of helmet violations near MG Road Junction between
   5–7 PM"), each with a probability gauge and a "notify patrol" mock action.
7. **System Performance Metrics** — pulls from your existing `evaluation.py`
   targets table (mAP, F1, precision, recall, plate accuracy, latency, FPR) —
   this page proves the system isn't just a demo, it has measurable targets.

---

## 6. New modules delivered in this package

| File | Module # | What it does | Status |
|---|---|---|---|
| `modules/offender_profiling.py` | 09 | Per-plate risk scoring (0–100), tiers, history | ✅ implemented & tested |
| `modules/hotspot_analytics.py` | 10 | Location ranking, heatmap points, time trends, peak periods | ✅ implemented & tested |
| `modules/risk_intelligence.py` | 11 | Composite Area Risk Score, ranking | ✅ implemented & tested |
| `modules/predictive_analytics.py` | 12 | Hour-of-week forecasting, alert generation | ✅ implemented & tested |
| `modules/explainability.py` | 13 | Reason templates + disposition rationale | ✅ implemented & tested |
| `modules/evidence.py` (patched) | 06 | `verify_integrity()` + auto-attached explanation | ✅ implemented & tested |
| `api.py` (patched) | — | 13 new endpoints wired to the above | ✅ implemented |
| `config/config.yaml` (patched) | — | Tunable weights/thresholds for all 4 new modules | ✅ implemented |

All five new modules were run end-to-end against your real demo evidence
records during this upgrade (18 sample violations, 1 repeat plate) — risk
scoring, hotspot ranking, area risk, and forecasting all produced correct,
sane output, and the evidence tamper check correctly flagged a modified file.

**Scoring logic, in plain terms** (so you can explain it to judges without
notes):
- **Vehicle Risk Score** — each violation contributes `severity_weight ×
  recency_decay`; scores are summed and passed through a saturating curve
  (`100 × (1 − e^(−sum/k))`) so frequent+severe+recent offenders climb toward
  100 while a single old minor violation stays low. A small bonus is added
  per *distinct* violation type, because a driver who commits 5 different
  kinds of violations is a different risk profile than one who's been caught
  parking illegally 5 times.
- **Area Risk Score** — weighted blend of 4 normalized components (volume
  35%, severity 25%, repeat-offender share 20%, traffic density 20%) —
  weights live in `config.yaml`, not hardcoded, so you can defend the choice
  live.
- **Predictive alerts** — built from an hour-of-week frequency table per
  (camera, violation type), normalized to the camera's own peak hour. This is
  intentionally simple (no black-box model) and trivially explainable: "this
  alert fires because helmet violations at this junction have historically
  peaked at this hour on this day of the week."

---

## 7. Data flow (end to end)

```
Camera frame
   → Preprocessing → Detection (+ByteTrack ID) → Violation Classification
   → LPR → Evidence Generation
        → writes ViolationRecord (+ explanation) to `violations` table
        → annotated PNG + SHA-256 hash to evidence/

[scheduled, async — not in the hot path]
   `violations` table
   → OffenderProfiler.recompute_all()      → offender_risk_scores
   → AreaRiskScorer.compute()              → area_risk_scores
   → PredictiveAnalyzer.forecast_next_hours() → predictions

Dashboard / API consumers read from the derived tables for instant response —
no recomputation happens on the request path.
```

This separation (write-path stays exactly as fast as your current prototype;
intelligence is computed off the hot path) is itself a talking point: it
shows you understand the difference between an inference pipeline and an
analytics platform.

---

## 8. Implementation roadmap (realistic for a hackathon)

| Phase | Hours | Tasks |
|---|---|---|
| 0 | 0–2 | Drop in the 4 new modules + patched `api.py`/`evidence.py`/`config.yaml` (already done in this package). Run `pipeline.py --demo` and confirm `violations` table populates. |
| 1 | 2–6 | Register 3–5 demo cameras via `/cameras` with real-looking junction names + GPS so the heatmap and hotspot pages aren't empty. Seed enough synthetic violations (reuse/extend the existing demo generator) across a few plates/cameras/hours to make offender + hotspot + predictive output meaningful. |
| 2 | 6–14 | Build the React dashboard shell (7 routes) consuming the 19 endpoints. Prioritize: Violations Feed → Repeat Offenders → Hotspot Analysis (map) — these are the most visually persuasive to judges. |
| 3 | 14–20 | Wire scheduled recompute (APScheduler is fastest: 10 lines in `pipeline.py` or a small `scheduler.py`). Add the explain-panel and evidence-verify button to the Violations Feed page. |
| 4 | 20–26 | Risk Intelligence + Predictive Analytics pages. Polish: risk-tier color coding, alert cards, component breakdown chart. |
| 5 | 26–32 | Smart Reporting: extend `analytics.py`'s existing PDF generator to add a "Top Offenders," "High-Risk Locations," and "Trend Analysis" section using data from the new tables. Daily/weekly/monthly variants are just different `days=` parameters. |
| 6 | 32–36 | Dockerize (`docker-compose`: api + postgres + redis + react), record demo video, rehearse pitch (§9). |

If time is short, cut Phase 5 (reporting) and Phase 6 polish first — Phases
0–4 are what make the judge-facing demo land.

---

## 9. Hackathon presentation points

**Opening line that reframes the pitch:** "Most traffic-violation systems stop
at detection. We built the layer that turns detections into *decisions* — who
to penalize, where to deploy patrols, and when violations are about to spike."

Suggested 5-slide arc:
1. **Problem** — traffic enforcement is reactive; cameras catch violations
   one at a time with no memory, no spatial intelligence, no foresight.
2. **What we built** — the original 9-module detection pipeline (keep this
   slide brief, it's your foundation, not your headline) → 4 new
   intelligence layers (offender risk, hotspot, area risk, predictive).
3. **Live demo** — Violations Feed → click a record → show the explanation
   panel → click "Verify Evidence" → jump to Repeat Offenders → jump to the
   heatmap → show a predictive alert firing.
4. **Why it's real, not hand-wavy** — show the scoring formulas, show the
   config file with tunable weights, show the evidence hash mismatch when you
   tamper with a file live on stage.
5. **Roadmap to production** — YOLOv11/RT-DETR + ByteTrack + PaddleOCR +
   PostgreSQL/TimescaleDB + Redis + Kafka + Kubernetes (you already have this
   table in your README; reuse it).

---

## 10. Judge-facing innovation highlights

- **Decision-support, not just detection** — the same raw violation feed now
  answers "who," "where," and "when next," which is what a traffic
  authority actually needs to act.
- **Explainability by default** — every auto-enforced ticket carries a
  reason string and supporting detections, directly addressing the
  "AI black box can't be used for legal enforcement" objection judges will
  raise.
- **Tamper-evident evidence with a live verify button** — most hackathon
  CV projects store a hash and never use it again; you can demonstrate
  detection of a modified file on stage in 10 seconds.
- **Statistically honest forecasting** — no overclaiming "AI predicts
  crime"; it's a transparent hour-of-week frequency model, explainable in
  one sentence, with a clear upgrade path to Prophet/Poisson-GLM.
- **Config-driven scoring** — severity weights and risk-component weights
  live in YAML, not buried in code, so the system is auditable and tunable
  by a non-engineer (a real requirement for government deployment).
- **Built on your existing schema** — zero breaking changes to the
  `violations` table or evidence pipeline; this is an additive intelligence
  layer, which is exactly how you'd ship this incrementally in production.

---

## 11. What makes this different from a standard "YOLO + OCR" project

| Standard hackathon traffic project | This platform |
|---|---|
| Detects violation, shows bounding box | Detects, scores the vehicle's risk, scores the location's risk, predicts recurrence |
| Confidence score only | Confidence + plain-English reason + supporting evidence + disposition rationale |
| One dashboard table | 7 purpose-built pages mapped to 7 distinct operational questions |
| Evidence image saved and forgotten | Evidence image with a callable, demoable integrity check |
| No memory across frames/sessions | Persistent per-vehicle and per-location profiles that compound over time |
| "We could add analytics later" | Analytics is a first-class, scheduled, decoupled layer — not an afterthought |
