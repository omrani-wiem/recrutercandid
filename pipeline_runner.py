"""
Orchestration des pauses/reprises du graphe LangGraph pour un couple (candidat, offre).

Le graphe (graph.py) se met en pause (interrupt()) à deux endroits :
  1. decision_review : score en zone grise -> l'humain choisit shortlist/reject
  2. email_review    : avant tout envoi d'e-mail de refus -> l'humain approuve/modifie/annule

Ce module fait le pont entre ces pauses et l'interface Streamlit : il lance/relance le
graphe, détecte les pauses, et renvoie un statut normalisé facile à afficher.

Limite connue : le checkpointer du graphe (MemorySaver) est en mémoire uniquement. Si
l'application redémarre pendant qu'un candidat est "en attente de décision", l'état
du graphe pour ce candidat est perdu (mais son score/explication, déjà sauvegardés
dans data/candidates.json, restent disponibles) — il faudra relancer l'analyse pour
ce candidat.
"""
from __future__ import annotations
from typing import Any

from langgraph.types import Command

STATUS_SHORTLISTED = "shortlisted"
STATUS_PENDING_DECISION = "pending_decision"
STATUS_PENDING_EMAIL_REVIEW = "pending_email_review"
STATUS_REJECTED_EMAIL_SENT = "rejected_email_sent"
STATUS_REJECTED_EMAIL_CANCELLED = "rejected_email_cancelled"
STATUS_REJECTED_EMAIL_FAILED = "rejected_email_failed"
STATUS_ERROR = "error"


def _thread_config(candidate_id: str, job_id: str) -> dict:
    return {"configurable": {"thread_id": f"{candidate_id}::{job_id}"}}


def _extract_interrupt(result: dict):
    """Renvoie le payload du premier interrupt en cours, ou None si le graphe a terminé."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def _to_plain(value: Any) -> Any:
    """Convertit un objet Pydantic (ou None) en dict simple, sinon renvoie tel quel."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _normalize(result: dict) -> dict:
    """Transforme l'état brut renvoyé par le graphe en un résultat normalisé pour l'UI.

    Le résultat est systématiquement sérialisé en dictionnaires simples (pas d'objets
    Pydantic), pour pouvoir être stocké tel quel dans data/candidates.json et relu sans
    code de désérialisation séparé.
    """
    interrupt_payload = _extract_interrupt(result)

    base = {
        "score": _to_plain(result.get("score")),
        "resume": _to_plain(result.get("resume")),
        "job": _to_plain(result.get("job")),
        "explanation": result.get("explanation", ""),
        "ocr_error": result.get("ocr_error"),
        "security_flags": result.get("security_flags", []),
        "rejection_email_draft": result.get("rejection_email_draft"),
        "rejection_email_sent": result.get("rejection_email_sent"),
        "email_send_error": result.get("email_send_error"),
        "interrupt_payload": interrupt_payload,
    }

    if interrupt_payload is not None:
        if interrupt_payload.get("type") == "decision_review":
            base["status"] = STATUS_PENDING_DECISION
        elif interrupt_payload.get("type") == "email_review":
            base["status"] = STATUS_PENDING_EMAIL_REVIEW
        else:
            base["status"] = STATUS_ERROR
        return base

    # Pas d'interrupt : le graphe est allé jusqu'au bout.
    if result.get("rejection_email_draft") is None:
        # Jamais passé par la branche refus -> shortlist automatique ou zone grise
        # validée en shortlist.
        base["status"] = STATUS_SHORTLISTED
    elif result.get("rejection_email_sent"):
        base["status"] = STATUS_REJECTED_EMAIL_SENT
    elif result.get("email_send_error") == "Envoi annulé par le recruteur":
        base["status"] = STATUS_REJECTED_EMAIL_CANCELLED
    else:
        base["status"] = STATUS_REJECTED_EMAIL_FAILED

    return base


def run_analysis(graph_app, candidate: dict, job: dict) -> dict:
    """Lance (ou relance depuis zéro) l'analyse d'un candidat pour une offre."""
    config = _thread_config(candidate["id"], job["id"])
    try:
        result = graph_app.invoke(
            {"cv_path": candidate["stored_path"], "jd_text": job["description"], "domain": job.get("domain")},
            config=config,
        )
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)


def resume_decision(graph_app, candidate: dict, job: dict, decision: str) -> dict:
    """Reprend un graphe en pause sur decision_review avec 'shortlist' ou 'reject'."""
    config = _thread_config(candidate["id"], job["id"])
    try:
        result = graph_app.invoke(Command(resume={"decision": decision}), config=config)
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)


def resume_email_review(
    graph_app,
    candidate: dict,
    job: dict,
    action: str,
    edited_subject: str | None = None,
    edited_body: str | None = None,
) -> dict:
    """
    Reprend un graphe en pause sur email_review.
    action: "approve" | "edit" | "cancel"
    """
    config = _thread_config(candidate["id"], job["id"])
    resume_payload: dict[str, Any] = {"action": action}
    if action == "edit":
        resume_payload["edited_draft"] = {"subject": edited_subject or "", "body": edited_body or ""}

    try:
        result = graph_app.invoke(Command(resume=resume_payload), config=config)
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)
