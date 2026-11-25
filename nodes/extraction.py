"""
Nœud LLM (un seul appel, pas d'agent) : transforme le texte du CV en ResumeProfile.
C'est le SEUL nœud du pipeline qui voit le texte du CV (masqué par anonymize_node) :
une fois passé ce nœud, plus aucun autre appel LLM n'y a accès.
"""
from __future__ import annotations
import json
import re

from dotenv import load_dotenv
from langgraph.types import interrupt

from core.state import ResumeProfile, SecurityFlag
from core.llm_client import client, MODEL_NAME
from security.prompt_injection_guard import (
    pre_llm_hook, post_llm_hook, escalate_if_unsafe,
)
from prompts.system_prompts import (
    CV_EXTRACTION_SYSTEM_PROMPT,
    CV_EXTRACTION_FEWSHOT,
    build_messages,
    extract_json_payload,
)

load_dotenv()

EXPECTED_KEYS = ["full_name", "skills", "years_experience", "education", "certifications"]

_TITLE_WORDS = re.compile(
    r"(?i)\b(étudiant|étudiante|etudiant|etudiante|ingénieur|ingenieur|développeur|"
    r"developpeur|stagiaire|consultant|responsable|manager|analyste|chef|master|"
    r"licence|llm|rag|cv|curriculum)\b"
)
_UNKNOWN = "INCONNU — échec extraction"


def clean_full_name(name: str) -> str:
    """Ne garde que prénom + nom : coupe aux séparateurs et aux mots de titre."""
    name = name or ""
    if "[" in name:                      # le LLM a recopié un marqueur [NOM]
        return ""
    name = re.split(r"[|·•,:(]|\s[-–—]\s", name)[0]   # garde les noms composés (Jean-Pierre)
    m = _TITLE_WORDS.search(name)
    if m:
        name = name[:m.start()]
    return " ".join(name.split()[:4]).strip()


def extraction_node(state: dict) -> dict:
    # Texte masqué si l'anonymisation a tourné, sinon texte brut.
    raw_text = state.get("anonymized_text") or state["extracted_text"]
    security_flags = list(state.get("security_flags", []))
    contact = state.get("contact_info") or {}

    # --- hook pré-LLM : classifieur anti-injection + sandwiching ----------------------
    guard = pre_llm_hook(
        untrusted_text=raw_text,
        task_instruction="Extrais les informations du CV ci-dessus au format JSON défini.",
        node_name="extraction",
        classify=True,
    )
    security_review_approved = False
    if not guard.is_safe:
        human_action = interrupt({
            "type": "security_review",
            "title": "Vérification du texte extrait du CV",
            "injection_probability": guard.injection_probability,
            "flags": guard.flags,
            "action_required": (
                "Vérifier le texte extrait dans l'interface, puis confirmer l'analyse "
                "ou l'annuler."
            ),
        })
        if human_action.get("action") != "approve":
            escalate_if_unsafe(guard, "extraction", security_flags)
            return {
                "security_review_approved": False,
                "security_review_rejected": True,
                "security_flags": security_flags,
            }

        security_review_approved = True
        security_flags.append(SecurityFlag(
            node="extraction",
            severity="warning",
            reason=(
                f"Classifieur anti-injection: probabilité {guard.injection_probability:.3f}; "
                "texte examiné et autorisé par le recruteur."
            ),
        ))

    messages = build_messages(
        CV_EXTRACTION_SYSTEM_PROMPT, CV_EXTRACTION_FEWSHOT, guard.sanitized_text
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1300,
        temperature=0,
        messages=messages,
    )
    output_text = response.choices[0].message.content
    if not isinstance(output_text, str):
        output_text = str(output_text)
    output_text = extract_json_payload(output_text)

    # --- hook post-LLM : JSON valide + clés attendues ---------------------------------
    post_guard = post_llm_hook(output_text, expected_schema_keys=EXPECTED_KEYS)
    if not post_guard.is_safe:
        security_flags.append(SecurityFlag(
            node="extraction_output",
            severity="warning",
            reason="; ".join(post_guard.flags),
        ))
        return {
            "resume": ResumeProfile(full_name=_UNKNOWN),
            "security_flags": security_flags,
            "security_review_approved": security_review_approved,
            "security_review_rejected": False,
        }

    try:
        data = json.loads(output_text)
        resume = ResumeProfile(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        security_flags.append(SecurityFlag(
            node="extraction",
            severity="warning",
            reason=f"échec de parsing JSON: {e}",
        ))
        resume = ResumeProfile(full_name=_UNKNOWN)
    else:
        # Identité réinjectée APRÈS le LLM (il ne l'a jamais vue grâce à l'anonymisation).
        name = clean_full_name(contact.get("full_name") or "") or clean_full_name(resume.full_name)
        resume.full_name = name or _UNKNOWN
        if contact.get("email"):
            resume.email = contact["email"]

    return {
        "resume": resume,
        "security_flags": security_flags,
        "security_review_approved": security_review_approved,
        "security_review_rejected": False,
    }