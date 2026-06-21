# ATVDCS – Automated Traffic Violation Detection & Classification System

> Computer Vision · Deep Learning · Edge Deployment  
> Solution Framework v1.0 — June 2026

---

## Overview

ATVDCS is an 8-module AI pipeline that processes traffic camera images end-to-end — from raw capture to tamper-evident evidence package — with minimal human intervention.

| Stage | Module | File |
|-------|--------|------|
| 01 | Image Preprocessing | `modules/preprocessing.py` |
| 02 | Vehicle & Road User Detection | `modules/detection.py` |
| 03 & 04 | Violation Detection & Classification | `modules/violation_classifier.py` |
| 05 | License Plate Recognition | `modules/lpr.py` |
| 06 | Evidence Generation | `modules/evidence.py` |
| 07 | Analytics & Reporting | `modules/analytics.py` |
| 08 | Performance Evaluation | `modules/evaluation.py` |

---

## Quick Start (CPU-only)

### 1. Install dependencies

```bash
cd atvdcs
pip install -r requirements.txt
```

For the minimal CPU-only prototype (no GPU, no PaddleOCR):
```bash
pip install opencv-python numpy pyyaml ultralytics matplotlib seaborn pandas fastapi uvicorn reportlab
```

### 2. Run the demo (no images or GPU needed)

```bash
python pipeline.py --demo
```

### 3. Process a real image

```bash
python pipeline.py --image /path/to/traffic_photo.jpg --camera CAM_01 --red
```

### 4. Process a directory of images

```bash
python pipeline.py --dir /path/to/images/ --camera CAM_01 --report
```

### 5. Start the REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs for interactive Swagger UI
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Violation Types Detected

| # | Type | Trigger |
|---|------|---------|
| 1 | Helmet Non-Compliance | Motorcycle rider, no helmet (confidence > 0.85) |
| 2 | Seatbelt Non-Compliance | Car driver, no visible seatbelt |
| 3 | Triple Riding | Motorcycle with 3+ person detections |
| 4 | Wrong-Side Driving | Vehicle heading against lane flow |
| 5 | Stop-Line Violation | Vehicle crosses stop line on RED |
| 6 | Red-Light Violation | Vehicle in intersection on RED |
| 7 | Illegal Parking | Vehicle stationary > 120s in restricted zone |

---

## Confidence Routing

| Score | Disposition | Action |
|-------|-------------|--------|
| ≥ 0.90 | `auto_enforce` | Ticket generated automatically |
| 0.70 – 0.90 | `secondary_check` | Second automated pass |
| < 0.70 | `human_review` | Routed to officer review queue |

---

## Output Files

```
evidence/
  <RECORD_ID>_annotated.png   ← lossless annotated frame
  <RECORD_ID>_preview.jpg     ← compressed preview
  <RECORD_ID>_record.json     ← signed metadata + SHA-256 hash

reports/
  trend_<ts>.png              ← daily violation trend chart
  dist_<ts>.png               ← violation type distribution
  conf_hist_<ts>.png          ← confidence score histogram
  report_<ts>.pdf             ← daily summary PDF

atvdcs.db                     ← SQLite violation database (swap for PostgreSQL)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process` | Upload image, get violation analysis |
| `GET` | `/violations` | Query violation records |
| `GET` | `/stats` | Summary statistics |
| `GET` | `/report/pdf` | Download daily PDF report |
| `GET` | `/export/csv` | Export violations as CSV |
| `GET` | `/health` | Liveness check |

---

## Production Upgrades

| Component | Prototype | Production |
|-----------|-----------|------------|
| Detection | YOLOv8n (mock fallback) | YOLOv8x / RT-DETR |
| Violation classifiers | Heuristics | EfficientNet-B4 fine-tuned |
| OCR | PaddleOCR / mock | TrOCR + PaddleOCR ensemble |
| Database | SQLite | PostgreSQL + TimescaleDB |
| Message queue | — | Apache Kafka |
| Inference | CPU | NVIDIA TensorRT (GPU) |
| Orchestration | — | Kubernetes + Helm |
| MLOps | — | MLflow + DVC |
| Monitoring | logs | Prometheus + Grafana |

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Object detection mAP@0.50 | > 0.92 |
| Violation F1 (weighted) | > 0.91 |
| Violation Precision | > 0.93 |
| Plate-level OCR accuracy | > 0.93 |
| End-to-end accuracy | > 0.88 |
| Latency (GPU) | < 400 ms |
| False positive rate | < 2% |

---

## Project Structure

```
atvdcs/
├── config/
│   └── config.yaml          ← all tuneable parameters
├── modules/
│   ├── preprocessing.py     ← M01
│   ├── detection.py         ← M02
│   ├── violation_classifier.py  ← M03 & M04
│   ├── lpr.py               ← M05
│   ├── evidence.py          ← M06
│   ├── analytics.py         ← M07
│   └── evaluation.py        ← M08
├── tests/
│   └── test_pipeline.py
├── pipeline.py              ← orchestrator + CLI
├── api.py                   ← FastAPI REST server
├── requirements.txt
└── README.md
```
