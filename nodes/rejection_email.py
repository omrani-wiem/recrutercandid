"""
Génère un brouillon d'e-mail de refus sans l'envoyer. L'envoi passe par l'intégration Resend
de graph.py uniquement après approbation via le nœud d'interruption human_review_email.
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from core.llm_client import client, MODEL_NAME



from prompts.system_prompts import (
    REJECTION_EMAIL_SYSTEM_PROMPT, REJECTION_EMAIL_FEWSHOT, build_messages, extract_json_payload
)
from security.prompt_injection_guard import (
    pre_llm_hook, post_llm_hook
)

load_dotenv()


def _parse_email_draft(raw_text: str) -> dict:
    """Parse a JSON email draft, even when the model wraps it in fences or extra text."""
    cleaned = extract_json_payload(raw_text)
    if not cleaned:
        return {
            "subject": "Suite à votre candidature",
            "body": "Bonjour, merci pour votre candidature. Nous reviendrons vers vous si votre profil correspond au poste.",
        }
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "subject": "Suite à votre candidature",
            "body": "Bonjour, merci pour votre candidature. Nous reviendrons vers vous si votre profil correspond au poste.",
        }


def rejection_email_node(state: dict) -> dict:
    resume, job, score = state["resume"], state["job"], state["score"]

    # Structured fields can still contain text originating from untrusted documents.
    context = (
        f"Candidat: {resume.full_name}, poste: {job.title}. "
        f"Compétences déjà maîtrisées par le candidat: {resume.skills or 'non renseignées'}. "
        f"Compétences manquantes pour ce poste: {score.missing_skills or 'aucune compétence technique majeure'}. "
        f"Filtres non satisfaits: {score.failed_filters or 'aucun'}. "
        f"Expérience du candidat: {resume.years_experience} ans "
        f"(minimum requis: {job.min_years_experience} ans)."
    )
    guard =  pre_llm_hook(
        context,
        "Rédige un brouillon d'email de refus factuel à partir des données fournies.",
        "rejection_email",
        classify=False,
    )


    messages = build_messages(
        REJECTION_EMAIL_SYSTEM_PROMPT, REJECTION_EMAIL_FEWSHOT, guard.sanitized_text
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        temperature=0.4,
        messages=messages,
    )

    draft_text = response.choices[0].message.content
    if not isinstance(draft_text, str):
        draft_text = str(draft_text)
    draft_text = extract_json_payload(draft_text)

    post_guard = post_llm_hook(draft_text, expected_schema_keys=["subject", "body"])
    if post_guard.is_safe:
        draft = json.loads(draft_text)
    else:
        # Repli : brouillon par défaut (le recruteur relit de toute façon l'email).
        draft = _parse_email_draft("")
    return {
        "rejection_email_draft": draft,
        "rejection_email_approved": False,
        "rejection_email_sent": False,
    }
