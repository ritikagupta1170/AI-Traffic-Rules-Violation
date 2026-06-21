"""
Module 13 – Explainable AI Layer
────────────────────────────────────────────────────────────────────────────────
Wraps every ViolationEvent with a human-readable explanation so officers,
auditors, and judges can see *why* the model flagged a violation — not just
a confidence number. Non-invasive: reads the `meta` dict and detection
objects the existing detectors already populate (helmet_score, rider_count,
signal_state, stationary_seconds) — no changes to detector logic required.

Wire-in (one line) inside modules/evidence.py, in `_create_evidence_package`,
right after the record is built:

    from modules.explainability import explain_violation
    record.metadata["explanation"] = explain_violation(event)

The explanation dict is then stored in ViolationRecord.metadata and surfaced
verbatim by the /violations/{id}/explain API endpoint.
"""

from typing import Any, Dict, List


# Per-type reason templates. {field} placeholders are filled from event.meta
# or from supporting_detections counts — falls back gracefully if a key is
# missing (e.g. prototype heuristic vs. production classifier wired in later).
_REASON_TEMPLATES: Dict[str, str] = {
    "helmet_non_compliance":
        "Rider head region detected on motorcycle; helmet-presence classifier "
        "score {helmet_score:.0%} (below {threshold:.0%} threshold) — helmet object absent.",
    "seatbelt_non_compliance":
        "Car driver detected; seatbelt-presence classifier did not find a visible "
        "belt strap across the torso region.",
    "triple_riding":
        "{rider_count} person detections overlapping a single motorcycle bounding "
        "box, exceeding the configured limit of 2 riders.",
    "wrong_side_driving":
        "Vehicle heading vector opposes the registered lane-flow direction for "
        "this camera's calibrated zone.",
    "stop_line_violation":
        "Signal state was {signal_state} and the vehicle's bounding box crossed "
        "the calibrated stop-line polygon.",
    "red_light_violation":
        "Signal state was {signal_state} and the vehicle was located inside the "
        "intersection bounding region.",
    "illegal_parking":
        "Vehicle remained stationary for {stationary_seconds:.0f}s inside a "
        "restricted/no-parking zone (limit: configurable, default 120s).",
}

_DEFAULTS = {"threshold": 0.85, "helmet_score": 0.0, "rider_count": 0,
             "signal_state": "UNKNOWN", "stationary_seconds": 0.0}


def explain_violation(event) -> Dict[str, Any]:
    """
    Build a structured, judge-friendly explanation for a ViolationEvent.

    Returns:
        {
          "violation_type": "helmet_non_compliance",
          "confidence": 0.98,
          "reason": "Rider head region detected ... helmet object absent.",
          "supporting_detections": ["motorcycle (0.91)", "person (0.88)"],
          "disposition": "auto_enforce",
          "disposition_reason": "Confidence 0.98 >= auto-enforce threshold 0.90"
        }
    """
    vtype = event.violation_type.value if hasattr(event.violation_type, "value") else str(event.violation_type)
    template = _REASON_TEMPLATES.get(vtype, "Violation pattern matched configured detection rule.")

    fill = {**_DEFAULTS, **event.meta}
    try:
        reason = template.format(**fill)
    except (KeyError, ValueError):
        reason = template  # never let a formatting miss block evidence generation

    supporting: List[str] = []
    if event.offending_vehicle is not None:
        supporting.append(f"{event.offending_vehicle.class_name} ({event.offending_vehicle.confidence:.0%})")
    for det in getattr(event, "supporting_detections", []):
        supporting.append(f"{det.class_name} ({det.confidence:.0%})")

    disposition = event.disposition.value if hasattr(event.disposition, "value") else str(event.disposition)
    disposition_reason = _disposition_reason(event.confidence, disposition)

    return {
        "violation_type": vtype,
        "confidence": round(float(event.confidence), 4),
        "reason": reason,
        "supporting_detections": supporting,
        "disposition": disposition,
        "disposition_reason": disposition_reason,
    }


def _disposition_reason(confidence: float, disposition: str) -> str:
    if disposition == "auto_enforce":
        return f"Confidence {confidence:.0%} meets the auto-enforce threshold — ticket issued without manual review."
    if disposition == "secondary_check":
        return f"Confidence {confidence:.0%} falls in the secondary-check band — routed for an automated second pass."
    return f"Confidence {confidence:.0%} is below the auto-enforce threshold — routed to human officer review."


def explain_record(record_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Read-path helper: pull the precomputed explanation back out of a stored
    ViolationRecord.metadata blob (already JSON-decoded)."""
    return record_metadata.get("explanation", {
        "reason": "No explanation captured at generation time for this legacy record.",
    })
