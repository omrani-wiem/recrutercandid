"""LLM node that explains a score using checked, structured candidate/job data."""
from __future__ import annotations
from dotenv import load_dotenv
from core.llm_client import client, MODEL_NAME

from prompts.system_prompts import EXPLANATION_SYSTEM_PROMPT
from security.prompt_injection_guard import pre_llm_hook, post_llm_text_hook

load_dotenv()


def explanation_node(state: dict) -> dict:
    resume, job, score = state["resume"], state["job"], state["score"]

    context = (
        f"Candidat: {resume.years_experience} ans d'expérience, compétences: {resume.skills}.\n"
        f"Poste: {job.title}, requis: {job.required_skills}, min expérience: "
        f"{job.min_years_experience} ans.\n"
        f"Score: similarité sémantique={score.semantic_similarity}, "
        f"overlap compétences={score.skill_overlap_ratio}, "
        f"filtres durs passés={score.hard_filters_passed}, score final={score.final_score}.\n"
        f"Compétences manquantes: {score.missing_skills}. Filtres échoués: {score.failed_filters}."
    )
    guard = pre_llm_hook(
    context,
    "Explique les indicateurs et le score sans modifier la décision. Réponds uniquement en français.",
    "explanation",
    classify=False,
)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=0.2,
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT + "\n\nRéponds UNIQUEMENT en français. N'utilise jamais le chinois ni aucune autre langue."},
            {"role": "user", "content": guard.sanitized_text},
        ],
    )
    explanation = response.choices[0].message.content
    if not isinstance(explanation, str):
        explanation = str(explanation)

    post_guard = post_llm_text_hook(explanation)
    if not post_guard.is_safe:
        return {"explanation": "Explication indisponible."}
    return {"explanation": post_guard.sanitized_text}