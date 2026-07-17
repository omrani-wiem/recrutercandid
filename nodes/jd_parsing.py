"""Nœud LLM single-shot : offre d'emploi (texte) -> JobRequirements structuré."""
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
    JD_PARSING_SYSTEM_PROMPT,
    JD_PARSING_FEWSHOT,
    build_messages,
    extract_json_payload,
)
from state import JobRequirements

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")


def _normalize_job_payload(payload: dict) -> dict:
    """Ensure numeric fields are valid before Pydantic validation."""
    normalized = dict(payload)
    if "min_years_experience" not in normalized or normalized["min_years_experience"] in (None, ""):
        normalized["min_years_experience"] = 0
    else:
        try:
            normalized["min_years_experience"] = float(normalized["min_years_experience"])
        except (TypeError, ValueError):
            normalized["min_years_experience"] = 0
    return normalized


def jd_parsing_node(state: dict) -> dict:
    security_flags = state.get("security_flags", [])

    guard = pre_llm_hook(
        untrusted_text=state["jd_text"],
        task_instruction="Extrais la structure de l'offre ci-dessus au format JSON défini.",
        node_name="jd_parsing",
    )
    escalate_if_unsafe(guard, "jd_parsing", security_flags)

    messages = build_messages(JD_PARSING_SYSTEM_PROMPT, JD_PARSING_FEWSHOT, guard.sanitized_text)

    response = client.chat.complete(
        model=MODEL_NAME,
        max_tokens=800,
        temperature=0,
        messages=messages,
    )
    output_text = response.choices[0].message.content
    if not isinstance(output_text, str):
        output_text = str(output_text)
    output_text = extract_json_payload(output_text)
    post_guard = post_llm_hook(output_text, expected_schema_keys=["title", "required_skills"])
    escalate_if_unsafe(post_guard, "jd_parsing_output", security_flags)

    payload = json.loads(output_text)
    normalized_payload = _normalize_job_payload(payload)
    job = JobRequirements(**normalized_payload)
    return {"job": job, "security_flags": security_flags}
