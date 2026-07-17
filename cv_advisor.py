"""
Génère un rapport d'amélioration de CV pour une offre précise, à destination des étudiants.
Un seul appel LLM (pas de graphe LangGraph ici, pas besoin d'interruption humaine : c'est un
outil d'aide, pas une décision de recrutement).
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

from security.prompt_injection_guard import pre_llm_hook, post_llm_hook, escalate_if_unsafe
from prompts.system_prompts import (
    CV_IMPROVEMENT_SYSTEM_PROMPT, CV_IMPROVEMENT_FEWSHOT, build_messages, extract_json_payload,
)

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

EXPECTED_KEYS = ["skills_manquantes", "elements_a_retirer", "conseils_generaux", "points_forts_a_souligner"]


def generate_cv_improvement_report(cv_text: str, job_title: str, job_description: str) -> dict:
    """Renvoie un dict avec les 4 clés du schéma, ou un dict d'erreur si échec."""
    security_flags: list[dict] = []

    guard = pre_llm_hook(
        untrusted_text=cv_text,
        task_instruction="Compare ce CV à l'offre ci-dessous et produis un rapport d'amélioration.",
        node_name="cv_advisor",
    )
    escalate_if_unsafe(guard, "cv_advisor", security_flags)

    user_content = f"OFFRE: {job_title} - {job_description}\n\nCV:\n{guard.sanitized_text}"
    messages = build_messages(CV_IMPROVEMENT_SYSTEM_PROMPT, CV_IMPROVEMENT_FEWSHOT, user_content)

    try:
        response = client.chat.complete(
            model=MODEL_NAME, max_tokens=1200, temperature=0.3, messages=messages,
        )
        output_text = response.choices[0].message.content
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = extract_json_payload(output_text)

        post_guard = post_llm_hook(output_text, expected_schema_keys=EXPECTED_KEYS)
        escalate_if_unsafe(post_guard, "cv_advisor_output", security_flags)

        data = json.loads(output_text)
        return {
            "skills_manquantes": data.get("skills_manquantes", []),
            "elements_a_retirer": data.get("elements_a_retirer", []),
            "conseils_generaux": data.get("conseils_generaux", []),
            "points_forts_a_souligner": data.get("points_forts_a_souligner", []),
            "error": None,
        }
    except Exception as exc:
        return {
            "skills_manquantes": [], "elements_a_retirer": [], "conseils_generaux": [],
            "points_forts_a_souligner": [], "error": str(exc),
        }
    
from prompts.system_prompts import (
    CV_IMPROVEMENT_SYSTEM_PROMPT, CV_IMPROVEMENT_FEWSHOT,
    MOTIVATION_LETTER_SYSTEM_PROMPT, MOTIVATION_LETTER_FEWSHOT,
    build_messages, extract_json_payload,
)

LETTER_EXPECTED_KEYS = ["subject", "body"]


def generate_motivation_letter(cv_text: str, job_title: str, job_description: str, company: str = "") -> dict:
    security_flags: list[dict] = []
    guard = pre_llm_hook(
        untrusted_text=cv_text,
        task_instruction="Rédige une lettre de motivation à partir de ce CV pour l'offre ci-dessous.",
        node_name="cv_advisor_letter",
    )
    escalate_if_unsafe(guard, "cv_advisor_letter", security_flags)

    entreprise_txt = f" chez {company}" if company else ""
    user_content = f"OFFRE: {job_title}{entreprise_txt} - {job_description}\n\nCV:\n{guard.sanitized_text}"
    messages = build_messages(MOTIVATION_LETTER_SYSTEM_PROMPT, MOTIVATION_LETTER_FEWSHOT, user_content)

    try:
        response = client.chat.complete(model=MODEL_NAME, max_tokens=900, temperature=0.4, messages=messages)
        output_text = response.choices[0].message.content
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = extract_json_payload(output_text)

        post_guard = post_llm_hook(output_text, expected_schema_keys=LETTER_EXPECTED_KEYS)
        escalate_if_unsafe(post_guard, "cv_advisor_letter_output", security_flags)

        data = json.loads(output_text)
        return {"subject": data.get("subject", ""), "body": data.get("body", ""), "error": None}
    except Exception as exc:
        return {"subject": "", "body": "", "error": str(exc)}