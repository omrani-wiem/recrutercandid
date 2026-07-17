"""
Génère un brouillon d'e-mail de refus. Ce nœud ne fait QUE générer le texte — il n'a aucun
accès à l'outil d'envoi. L'envoi réel se fait exclusivement via mcp_servers/email_mcp_server.py,
appelé depuis graph.py APRÈS le passage par le nœud d'interruption human_review_email.
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

from prompts.system_prompts import (
    REJECTION_EMAIL_SYSTEM_PROMPT, REJECTION_EMAIL_FEWSHOT, build_messages, extract_json_payload
)

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")


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

    # entrée déjà structurée (pas de texte brut candidat) -> pas de sandwiching nécessaire ici,
    # mais on garde le principe : seules les données validées/structurées alimentent ce nœud
    context = (
        f"Candidat: {resume.full_name}, poste: {job.title}. "
        f"Compétences déjà maîtrisées par le candidat: {resume.skills or 'non renseignées'}. "
        f"Compétences manquantes pour ce poste: {score.missing_skills or 'aucune compétence technique majeure'}. "
        f"Filtres non satisfaits: {score.failed_filters or 'aucun'}. "
        f"Expérience du candidat: {resume.years_experience} ans "
        f"(minimum requis: {job.min_years_experience} ans)."
    )

    messages = build_messages(REJECTION_EMAIL_SYSTEM_PROMPT, REJECTION_EMAIL_FEWSHOT, context)

    response = client.chat.complete(
        model=MODEL_NAME,
        max_tokens=600,
        temperature=0.4,
        messages=messages,
    )

    draft_text = response.choices[0].message.content
    if not isinstance(draft_text, str):
        draft_text = str(draft_text)
    draft = _parse_email_draft(draft_text)
    # IMPORTANT : rejection_email_approved reste TOUJOURS False ici. Il y avait
    # auparavant une auto-approbation (`bool(resume.email and os.getenv("RESEND_API_KEY"))`)
    # qui envoyait le mail automatiquement dès qu'un email candidat était détecté et
    # qu'une clé Resend était configurée — sans jamais qu'un humain ne le relise. C'est
    # exactement ce que le point d'arrêt human_review_email_node (nodes/human_review.py)
    # est censé empêcher. La seule façon d'approuver un envoi est une action humaine
    # explicite via ce point d'arrêt.
    return {
        "rejection_email_draft": draft,
        "rejection_email_approved": False,
        "rejection_email_sent": False,
    }
