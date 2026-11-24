"""
Génère un rapport d'amélioration de CV pour une offre précise, à destination des étudiants.
Un seul appel LLM (pas de graphe LangGraph ici, pas besoin d'interruption humaine : c'est un
outil d'aide, pas une décision de recrutement).
"""
from __future__ import annotations
import json
import os
from dotenv import load_dotenv
from core.llm_client import client, MODEL_NAME



from security.prompt_injection_guard import (
    pre_llm_hook, post_llm_hook,
)
from prompts.system_prompts import (
    CV_IMPROVEMENT_SYSTEM_PROMPT, CV_IMPROVEMENT_FEWSHOT, build_messages, extract_json_payload,
)

load_dotenv()

EXPECTED_KEYS = ["skills_manquantes", "elements_a_retirer", "conseils_generaux", "points_forts_a_souligner"]


def generate_cv_improvement_report(cv_text: str, job_title: str, job_description: str) -> dict:
    """Renvoie un dict avec les 4 clés du schéma, ou un dict d'erreur si échec."""
    security_flags: list[dict] = []

    try:
        guard = pre_llm_hook(
            untrusted_text=(
                f"OFFRE: {job_title}\n{job_description}\n\nCV:\n{cv_text}"
            ),
            task_instruction="Compare ce CV à l'offre ci-dessous et produis un rapport d'amélioration.",
            node_name="cv_advisor",
        )
        if not guard.is_safe:
           security_flags.append({"node": "cv_advisor", "reason": "; ".join(guard.flags)})
        messages = build_messages(
            CV_IMPROVEMENT_SYSTEM_PROMPT, CV_IMPROVEMENT_FEWSHOT, guard.sanitized_text
        )

        response = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=1200, temperature=0.3, messages=messages,
        )
        output_text = response.choices[0].message.content
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = extract_json_payload(output_text)

        post_guard = post_llm_hook(output_text, expected_schema_keys=EXPECTED_KEYS)
        if not post_guard.is_safe:
            return {
                "skills_manquantes": [], "elements_a_retirer": [], "conseils_generaux": [],
                "points_forts_a_souligner": [],
                "error": "Le modèle n'a pas renvoyé un rapport valide. Relance l'analyse.",
            }

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
    try:
        entreprise_txt = f" chez {company}" if company else ""
        guard = pre_llm_hook(
            untrusted_text=(
                f"OFFRE: {job_title}{entreprise_txt}\n{job_description}\n\nCV:\n{cv_text}"
            ),
            task_instruction="Rédige une lettre de motivation à partir de ce CV pour l'offre ci-dessous.",
            node_name="cv_advisor_letter",
        )
        if not guard.is_safe:
            security_flags.append({"node": "cv_advisor_letter", "reason": "; ".join(guard.flags)})
        messages = build_messages(
            MOTIVATION_LETTER_SYSTEM_PROMPT, MOTIVATION_LETTER_FEWSHOT,
            guard.sanitized_text,
        )

        response = client.chat.completions.create(model=MODEL_NAME, max_tokens=900, temperature=0.4, messages=messages)
        output_text = response.choices[0].message.content
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = extract_json_payload(output_text)

        post_guard = post_llm_hook(output_text, expected_schema_keys=LETTER_EXPECTED_KEYS)
        if not post_guard.is_safe:
            return {
                "subject": "", "body": "",
                "error": "Le modèle n'a pas renvoyé une lettre valide. Relance la génération.",
            }

        data = json.loads(output_text)
        return {"subject": data.get("subject", ""), "body": data.get("body", ""), "error": None}
    except Exception as exc:
        return {"subject": "", "body": "", "error": str(exc)}