"""
Stocke les avis du recruteur sur les candidats ("ce candidat était pertinent ou non")
pour permettre au calcul du score de se recalibrer avec le temps (voir offline_update
dans scoring_model.py). Sans ce module, il n'y avait aucun moyen de faire remonter un
avis recruteur jusqu'au calcul du score : offline_update() existait mais n'était
jamais appelée nulle part.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

FEEDBACK_LOG_PATH = Path(__file__).parent / "feedback_log.jsonl"


def append_feedback(features: dict, reward: float, candidate_name: str = "", job_title: str = "") -> None:
    """Enregistre un avis recruteur (reward=1.0 si le candidat était pertinent, 0.0 sinon)."""
    entry = {
        "features": features,
        "reward": reward,
        "candidate_name": candidate_name,
        "job_title": job_title,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with FEEDBACK_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_feedback_log() -> list[dict]:
    """Charge tous les avis en attente (pas encore utilisés pour recalibrer le score)."""
    if not FEEDBACK_LOG_PATH.exists():
        return []
    entries = []
    with FEEDBACK_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def clear_feedback_log() -> None:
    """Vide le journal une fois les avis pris en compte dans le recalibrage."""
    if FEEDBACK_LOG_PATH.exists():
        FEEDBACK_LOG_PATH.unlink()
