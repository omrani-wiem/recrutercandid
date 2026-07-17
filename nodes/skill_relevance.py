"""
Nœud LLM single-shot : juge si les compétences concrètes d'un candidat satisfont les
exigences d'une offre formulées en termes généraux (ex: "Python" satisfait "compétences
en analyse de données"). Ne décide jamais du score final ni d'un accept/reject — le
résultat (skill_relevance_judgment) est ensuite consommé par nodes/matching_scoring.py,
qui reste le seul endroit où le score est calculé, de façon déterministe et auditable.

Optionnel : si ce nœud échoue (LLM indisponible, réponse invalide), le pipeline continue
avec la correspondance mot-à-mot classique (voir matching_scoring._skill_overlap), sans
planter. C'est un enrichissement du signal, pas une dépendance dure.
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
    SKILL_RELEVANCE_SYSTEM_PROMPT,
    SKILL_RELEVANCE_FEWSHOT,
    build_messages,
    extract_json_payload,
)
from prompts.domain_prompts import get_domain_context

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")


def skill_relevance_node(state: dict) -> dict:
    """
    Ajoute state["skill_relevance_judgment"] si l'appel LLM réussit, sinon laisse ce champ
    à None : matching_scoring_node retombe alors sur la correspondance mot-à-mot classique.
    """
    security_flags = state.get("security_flags", [])
    resume = state.get("resume")
    job = state.get("job")

    if not resume or not job:
        return {"skill_relevance_judgment": None}

    resume_skills = getattr(resume, "skills", []) or []
    required_skills = getattr(job, "required_skills", []) or []
    nice_to_have_skills = getattr(job, "nice_to_have_skills", []) or []

    if not resume_skills or not (required_skills or nice_to_have_skills):
        return {"skill_relevance_judgment": None}

    untrusted_text = (
        f"Compétences du candidat: {json.dumps(resume_skills, ensure_ascii=False)}\n"
        f"Compétences requises: {json.dumps(required_skills, ensure_ascii=False)}\n"
        f"Compétences souhaitées: {json.dumps(nice_to_have_skills, ensure_ascii=False)}"
    )

    domain_context = get_domain_context(state.get("domain"))
    if domain_context:
        # Contexte métier ajouté APRÈS les données candidat/offre (données non fiables),
        # pour rester clairement distinct : c'est une aide au jugement, pas une donnée
        # candidat/offre à interpréter comme telle.
        untrusted_text += f"\n\nContexte du domaine (repère de vocabulaire) : {domain_context}"

    guard = pre_llm_hook(
        untrusted_text=untrusted_text,
        task_instruction="Juge la correspondance entre les deux listes de compétences ci-dessus.",
        node_name="skill_relevance",
    )
    escalate_if_unsafe(guard, "skill_relevance", security_flags)

    messages = build_messages(SKILL_RELEVANCE_SYSTEM_PROMPT, SKILL_RELEVANCE_FEWSHOT, guard.sanitized_text)

    try:
        response = client.chat.complete(
            model=MODEL_NAME,
            max_tokens=500,
            temperature=0,
            messages=messages,
        )
        output_text = response.choices[0].message.content
        if not isinstance(output_text, str):
            output_text = str(output_text)
        output_text = extract_json_payload(output_text)

        post_guard = post_llm_hook(output_text, expected_schema_keys=["matched_required", "missing_required"])
        escalate_if_unsafe(post_guard, "skill_relevance_output", security_flags)

        judgment = json.loads(output_text)
    except Exception:
        # Échec silencieux et non bloquant : le calcul du score retombe sur la
        # correspondance mot-à-mot classique (voir matching_scoring._skill_overlap).
        return {"skill_relevance_judgment": None, "security_flags": security_flags}

    return {"skill_relevance_judgment": judgment, "security_flags": security_flags}
