"""
Points d'arrêt du graphe. LangGraph pause l'exécution à ces nœuds (via interrupt()) et
attend une action humaine explicite avant de continuer — le graphe ne peut pas "deviner"
une approbation.

Deux interruptions distinctes et volontairement séparées :
  1. human_review_decision : cas de score en zone grise -> décision shortlist/reject
  2. human_review_email    : approbation OBLIGATOIRE avant tout envoi d'e-mail de refus,
                             même si la décision de rejet elle-même était déjà validée
"""
from __future__ import annotations
from langgraph.types import interrupt


def _get(obj, key: str, default=None):
    """Lit un champ sur un objet Pydantic ou sur un dict (selon la désérialisation)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    return obj.model_dump() if hasattr(obj, "model_dump") else {}


def human_review_decision_node(state: dict) -> dict:
    payload = {
        "type": "decision_review",
        "resume_name": _get(state["resume"], "full_name", ""),
        "job_title": _get(state["job"], "title", ""),
        "score": _to_dict(state["score"]),
        "explanation": state.get("explanation", ""),
        "security_flags": [_to_dict(f) for f in state.get("security_flags", [])],
        "action_required": "Choisir: 'shortlist' ou 'reject'",
    }
    human_decision = interrupt(payload)  # exécution suspendue ici jusqu'à reprise externe
    return {"decision": human_decision["decision"]}


def human_review_email_node(state: dict) -> dict:
    resume = state["resume"]
    payload = {
        "type": "email_review",
        "candidate_name": _get(resume, "full_name", ""),
        "recipient": _get(resume, "email", None),
        "draft": state["rejection_email_draft"],
        "action_required": "Approuver, modifier, ou annuler l'envoi",
    }
    human_action = interrupt(payload)
    action = human_action.get("action")

    if action == "approve":
        return {"rejection_email_approved": True,
                "rejection_email_draft": state["rejection_email_draft"]}
    if action == "edit":
        return {"rejection_email_approved": True,
                "rejection_email_draft": human_action["edited_draft"]}

    # Annulé (ou action inconnue) : ne sera jamais envoyé. La raison est marquée
    # explicitement pour que l'UI distingue une annulation d'un échec technique.
    return {"rejection_email_approved": False,
            "email_send_error": "Envoi annulé par le recruteur"}