"""
Nœud LLM (un seul appel, pas d'agent) : transforme le texte brut du CV en ResumeProfile.
C'est le SEUL nœud du pipeline qui voit le texte brut du CV — une fois passé ce nœud, plus
aucun autre appel LLM n'a accès au texte non structuré. Ça réduit drastiquement la surface
d'attaque du prompt injection au reste du graphe.
"""
from __future__ import annotations
import json
import importlib
import os
from openai import OpenAI
from dotenv import load_dotenv
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
def interrupt(payload: dict):
    """Load LangGraph's interrupt when a human review is actually required."""
    try:
        return importlib.import_module("langgraph.types").interrupt(payload)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Security review requires LangGraph; install the 'langgraph' package."
        ) from exc


load_dotenv()


EXPECTED_KEYS = ["full_name", "skills", "years_experience", "education", "certifications"]


def extraction_node(state: dict) -> dict:
    raw_text = state["extracted_text"]
    security_flags = state.get("security_flags", [])

    # --- hook pré-LLM : scan + sandwiching --------------------------------------------
    guard = pre_llm_hook(
        untrusted_text=raw_text,
        task_instruction="Extrais les informations du CV ci-dessus au format JSON défini.",
        node_name="extraction",
         classify=False,
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

        # --- hook post-LLM : vérifie que la sortie est bien un JSON conforme --------------
    post_guard = post_llm_hook(output_text, expected_schema_keys=EXPECTED_KEYS)
    if not post_guard.is_safe:
        security_flags.append(SecurityFlag(
            node="extraction_output",
            severity="warning",
            reason="; ".join(post_guard.flags),
        ))
        return {
            "resume": ResumeProfile(full_name="INCONNU — échec extraction"),
            "security_flags": security_flags,
            "security_review_approved": security_review_approved,
            "security_review_rejected": False,
        }

    try:
        data = json.loads(output_text)
        resume = ResumeProfile(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # échec de parsing = on ne devine jamais, on route vers la revue humaine
        security_flags.append(SecurityFlag(
            node="extraction",
            severity="warning",
            reason=f"échec de parsing JSON: {e}",
        ))
        resume = ResumeProfile(full_name="INCONNU — échec extraction")

    return {
        "resume": resume,
        "security_flags": security_flags,
        "security_review_approved": security_review_approved,
        "security_review_rejected": False,
    }