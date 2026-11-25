"""
Orchestration des pauses/reprises du graphe LangGraph pour un couple (candidat, offre).

Le graphe (graph.py) se met en pause (interrupt()) à trois endroits :
  1. security_review : contenu CV signalé -> le recruteur vérifie avant tout appel LLM
  2. decision_review : score en zone grise -> l'humain choisit shortlist/reject
  3. email_review    : avant tout envoi d'e-mail de refus -> l'humain approuve/modifie/annule

Ce module fait le pont entre ces pauses et l'interface Streamlit : il lance/relance le
graphe, détecte les pauses, et renvoie un statut normalisé facile à afficher.

Limite connue : le checkpointer du graphe (MemorySaver) est en mémoire uniquement. Si
l'application redémarre pendant qu'un candidat est "en attente de décision", l'état
du graphe pour ce candidat est perdu (mais son score/explication, déjà sauvegardés
dans data/candidates.json, restent disponibles) : il faudra relancer l'analyse.
"""
from __future__ import annotations
from typing import Any

from langgraph.types import Command

STATUS_SHORTLISTED = "shortlisted"
STATUS_PENDING_SECURITY_REVIEW = "pending_security_review"
STATUS_SECURITY_REVIEW_REJECTED = "security_review_rejected"
STATUS_PENDING_DECISION = "pending_decision"
STATUS_PENDING_EMAIL_REVIEW = "pending_email_review"
STATUS_REJECTED_EMAIL_SENT = "rejected_email_sent"
STATUS_REJECTED_EMAIL_CANCELLED = "rejected_email_cancelled"
STATUS_REJECTED_EMAIL_FAILED = "rejected_email_failed"
STATUS_ERROR = "error"

_EXPIRED_MESSAGE = (
    "Session d'analyse expirée (application redémarrée). "
    "Clique sur « Réinitialiser » puis relance l'analyse de ce candidat."
)

try:
    from langfuse.langchain import CallbackHandler   # langfuse 3.x
except ImportError:
    try:
        from langfuse.callback import CallbackHandler   # langfuse 2.x
    except ImportError:
        CallbackHandler = None


def _thread_config(candidate_id: str, job_id: str) -> dict:
    thread_id = f"{candidate_id}::{job_id}"
    config = {"configurable": {"thread_id": thread_id}}
    if CallbackHandler is not None:
        config["callbacks"] = [CallbackHandler()]
        config["metadata"] = {"langfuse_session_id": thread_id}
    return config


def _checkpoint_alive(graph_app, candidate: dict, job: dict) -> bool:
    """Vrai si le graphe de ce candidat est bien en pause (état encore en mémoire)."""
    try:
        snapshot = graph_app.get_state(_thread_config(candidate["id"], job["id"]))
    except Exception:
        return False
    return bool(snapshot.values) and bool(snapshot.next)


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
    Pydantic), pour pouvoir être stocké tel quel dans data/candidates.json.
    """
    interrupt_payload = _extract_interrupt(result)

    base = {
        "score": _to_plain(result.get("score")),
        "resume": _to_plain(result.get("resume")),
        "job": _to_plain(result.get("job")),
        "explanation": result.get("explanation", ""),
        "ocr_error": result.get("ocr_error"),
        "ocr_method": result.get("ocr_method"),
        "ocr_seconds": result.get("ocr_seconds"),
        "vlm_seconds": result.get("vlm_seconds"),
        "security_flags": [
            _to_plain(flag) for flag in result.get("security_flags", [])
        ],
        "rejection_email_draft": result.get("rejection_email_draft"),
        "rejection_email_sent": result.get("rejection_email_sent"),
        "email_send_error": result.get("email_send_error"),
        "email_sent_to": result.get("email_sent_to"),
        "email_test_mode": result.get("email_test_mode"),
        "email_original_recipient": result.get("email_original_recipient"),
        "interrupt_payload": interrupt_payload,
    }

    if interrupt_payload is not None:
        if interrupt_payload.get("type") == "decision_review":
            base["status"] = STATUS_PENDING_DECISION
        elif interrupt_payload.get("type") == "email_review":
            base["status"] = STATUS_PENDING_EMAIL_REVIEW
        elif interrupt_payload.get("type") == "security_review":
            base["status"] = STATUS_PENDING_SECURITY_REVIEW
        else:
            base["status"] = STATUS_ERROR
        return base

    # Pas d'interrupt : le graphe est allé jusqu'au bout (ou s'est arrêté).

    # Revue de sécurité refusée par le recruteur.
    if result.get("security_review_rejected"):
        base["status"] = STATUS_SECURITY_REVIEW_REJECTED
        return base

    # L'OCR de secours a échoué : jamais de statut « Retenu » sur un CV illisible.
    if result.get("ocr_method") == "qwen_vlm_failed" and not result.get("resume"):
        base["status"] = STATUS_ERROR
        base["error"] = result.get("ocr_error") or "Lecture du CV impossible."
        return base

    # Le graphe s'est terminé sans score : analyse incomplète, pas une shortlist.
    if result.get("score") is None:
        base["status"] = STATUS_ERROR
        base["error"] = "Analyse incomplète : aucun score n'a été calculé."
        return base

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
            {
                "cv_path": candidate["stored_path"],
                "jd_text": job["description"],
                "domain": job.get("domain"),
            },
            config=config,
        )
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)


def resume_decision(graph_app, candidate: dict, job: dict, decision: str) -> dict:
    """Reprend un graphe en pause sur decision_review avec 'shortlist' ou 'reject'."""
    if not _checkpoint_alive(graph_app, candidate, job):
        return {"status": STATUS_ERROR, "error": _EXPIRED_MESSAGE}

    config = _thread_config(candidate["id"], job["id"])
    try:
        result = graph_app.invoke(Command(resume={"decision": decision}), config=config)
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)


def security_review_text(graph_app, candidate: dict, job: dict) -> str:
    """Read the extracted CV text from the in-memory checkpoint for reviewer display."""
    try:
        snapshot = graph_app.get_state(_thread_config(candidate["id"], job["id"]))
    except Exception:
        return ""
    return snapshot.values.get("extracted_text", "")


def resume_security_review(
    graph_app,
    candidate: dict,
    job: dict,
    action: str,
) -> dict:
    """Resume a graph paused for a possible prompt injection in extracted CV text."""
    if action not in {"approve", "reject"}:
        return {"status": STATUS_ERROR, "error": "Action de revue de sécurité invalide."}

    if not _checkpoint_alive(graph_app, candidate, job):
        return {"status": STATUS_ERROR, "error": _EXPIRED_MESSAGE}

    config = _thread_config(candidate["id"], job["id"])
    try:
        result = graph_app.invoke(Command(resume={"action": action}), config=config)
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
    if not _checkpoint_alive(graph_app, candidate, job):
        return {"status": STATUS_ERROR, "error": _EXPIRED_MESSAGE}

    config = _thread_config(candidate["id"], job["id"])
    resume_payload: dict[str, Any] = {"action": action}
    if action == "edit":
        resume_payload["edited_draft"] = {
            "subject": edited_subject or "",
            "body": edited_body or "",
        }

    try:
        result = graph_app.invoke(Command(resume=resume_payload), config=config)
    except Exception as exc:
        return {"status": STATUS_ERROR, "error": str(exc)}
    return _normalize(result)