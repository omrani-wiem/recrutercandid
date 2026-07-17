"""
Nœud LLM (un seul appel, pas d'agent) : transforme le texte brut du CV en ResumeProfile.
C'est le SEUL nœud du pipeline qui voit le texte brut du CV — une fois passé ce nœud, plus
aucun autre appel LLM n'a accès au texte non structuré. Ça réduit drastiquement la surface
d'attaque du prompt injection au reste du graphe.
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
    CV_EXTRACTION_SYSTEM_PROMPT,
    CV_EXTRACTION_FEWSHOT,
    build_messages,
    extract_json_payload,
)
from state import ResumeProfile

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")

EXPECTED_KEYS = ["full_name", "skills", "years_experience", "education", "certifications"]


def extraction_node(state: dict) -> dict:
    raw_text = state["extracted_text"]
    security_flags = state.get("security_flags", [])

    # --- hook pré-LLM : scan + sandwiching --------------------------------------------
    guard = pre_llm_hook(
        untrusted_text=raw_text,
        task_instruction="Extrais les informations du CV ci-dessus au format JSON défini.",
        node_name="extraction",
    )
    escalate_if_unsafe(guard, "extraction", security_flags)

    messages = build_messages(
        CV_EXTRACTION_SYSTEM_PROMPT, CV_EXTRACTION_FEWSHOT, guard.sanitized_text
    )

    response = client.chat.complete(
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
    escalate_if_unsafe(post_guard, "extraction_output", security_flags)

    try:
        data = json.loads(output_text)
        resume = ResumeProfile(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # échec de parsing = on ne devine jamais, on route vers la revue humaine
        security_flags.append({
            "node": "extraction",
            "severity": "warning",
            "reason": f"échec de parsing JSON: {e}",
        })
        resume = ResumeProfile(full_name="INCONNU — échec extraction")

    return {"resume": resume, "security_flags": security_flags}
