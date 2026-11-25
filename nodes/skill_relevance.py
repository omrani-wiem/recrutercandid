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
from core.llm_client import client, MODEL_NAME
from core.state import SecurityFlag
from security.prompt_injection_guard import UnsafeLLMContentError, pre_llm_hook, post_llm_hook



from security.prompt_injection_guard import (
     pre_llm_hook, post_llm_hook,
)

from prompts.system_prompts import (
    SKILL_RELEVANCE_SYSTEM_PROMPT,
    SKILL_RELEVANCE_FEWSHOT,
    build_messages,
    extract_json_payload,
)
from prompts.domain_prompts import get_domain_context

load_dotenv()
def _sanitize_judgment(judgment: dict, resume_skills, required, nice) -> dict:
    """Un nom d'outil précis ne peut être « en commun » que si le candidat l'a littéralement."""
    have = {s.strip().lower() for s in resume_skills}

    def literally_has(skill: str) -> bool:
        s = skill.strip().lower()
        return any(s == h or s in h or h in s for h in have)

    def is_specific(skill: str) -> bool:
        # un nom court (1 à 2 mots) = un outil ou une techno précise ; une phrase = exigence générale
        return len(skill.split()) <= 2

    def clean(matched, universe):
        universe_l = {u.lower(): u for u in universe}
        out = []
        for m in matched or []:
            ref = universe_l.get(str(m).strip().lower())
            if ref is None:
                continue                      # le LLM a inventé une exigence
            if is_specific(ref) and not literally_has(ref):
                continue                      # outil précis non possédé : rejeté
            out.append(ref)
        return out

    matched_req = clean(judgment.get("matched_required"), required)
    judgment["matched_required"] = matched_req
    judgment["missing_required"] = [r for r in required if r not in matched_req]
    if "matched_nice_to_have" in judgment:
        judgment["matched_nice_to_have"] = clean(judgment["matched_nice_to_have"], nice)
    return judgment

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
        untrusted_text += f"\n\nContexte du domaine (repère de vocabulaire) : {domain_context}"

        guard = pre_llm_hook(
        untrusted_text=untrusted_text,
        task_instruction="Juge la correspondance entre les deux listes de compétences ci-dessus.",
        node_name="skill_relevance",
        classify=False,
    )
    if not guard.is_safe:
        security_flags.append(SecurityFlag(
            node="skill_relevance",
            severity="warning",
            reason="; ".join(guard.flags) + " — jugement LLM ignoré, repli mot-à-mot.",
        ))
        return {"skill_relevance_judgment": None, "security_flags": security_flags}
    messages = build_messages(SKILL_RELEVANCE_SYSTEM_PROMPT, SKILL_RELEVANCE_FEWSHOT, guard.sanitized_text)

    try:
        response = client.chat.completions.create(
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
        if not post_guard.is_safe:
            security_flags.append(SecurityFlag(
                node="skill_relevance_output",
                severity="warning",
                reason="; ".join(post_guard.flags),
            ))
            return {"skill_relevance_judgment": None, "security_flags": security_flags}

        judgment = json.loads(output_text)
        judgment = _sanitize_judgment(judgment, resume_skills, required_skills, nice_to_have_skills)
    except UnsafeLLMContentError:
        raise
    except Exception:
        # Échec silencieux et non bloquant : le calcul du score retombe sur la
        # correspondance mot-à-mot classique (voir matching_scoring._skill_overlap).
        return {"skill_relevance_judgment": None, "security_flags": security_flags}

    return {"skill_relevance_judgment": judgment, "security_flags": security_flags}
