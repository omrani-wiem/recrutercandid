"""
Stocke les avis du recruteur sur les candidats ("ce candidat était pertinent ou non")
pour permettre le recalibrage des poids du score à partir du feedback recruteur (voir
calibrate_scoring_weights dans scoring_model.py).
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

FEEDBACK_LOG_PATH = Path(__file__).parent / "feedback_log.jsonl"
LEGACY_FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent / "rl" / "feedback_log.jsonl"


def append_feedback(
    features: dict,
    feedback_label: float,
    candidate_name: str = "",
    job_title: str = "",
) -> None:
    """Enregistre un avis binaire (1.0 si pertinent, 0.0 sinon)."""
    entry = {
        "features": features,
        "feedback_label": feedback_label,
        "candidate_name": candidate_name,
        "job_title": job_title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_feedback_log() -> list[dict]:
    """Charge tous les avis en attente (pas encore utilisés pour recalibrer le score)."""
    entries = []
    for path in (LEGACY_FEEDBACK_LOG_PATH, FEEDBACK_LOG_PATH):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries


def clear_feedback_log() -> None:
    """Vide les journaux de feedback une fois les avis pris en compte."""
    for path in (LEGACY_FEEDBACK_LOG_PATH, FEEDBACK_LOG_PATH):
        if path.exists():
            path.unlink()
