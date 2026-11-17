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


def human_review_decision_node(state: dict) -> dict:
    payload = {
        "type": "decision_review",
        "resume_name": state["resume"].full_name,
        "job_title": state["job"].title,
        "score": state["score"].model_dump(),
        "explanation": state.get("explanation", ""),
        "security_flags": state.get("security_flags", []),
        "action_required": "Choisir: 'shortlist' ou 'reject'",
    }
    human_decision = interrupt(payload)  # exécution suspendue ici jusqu'à reprise externe
    return {"decision": human_decision["decision"]}


def human_review_email_node(state: dict) -> dict:
    payload = {
        "type": "email_review",
        "recipient": state["resume"].full_name,
        "draft": state["rejection_email_draft"],
        "action_required": "Approuver, modifier, ou annuler l'envoi",
    }
    human_action = interrupt(payload)

    if human_action["action"] == "approve":
        return {"rejection_email_approved": True,
                "rejection_email_draft": state["rejection_email_draft"]}
    if human_action["action"] == "edit":
        return {"rejection_email_approved": True,
                "rejection_email_draft": human_action["edited_draft"]}
    # annulé par le recruteur : ne sera jamais envoyé. On marque explicitement la
    # raison (plutôt que de laisser email_send_error à None) pour que l'UI distingue
    # une annulation volontaire d'un échec technique d'envoi.
    return {"rejection_email_approved": False, "email_send_error": "Envoi annulé par le recruteur"}
