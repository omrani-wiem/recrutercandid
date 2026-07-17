"""
Nœud LLM : génère une explication lisible du score. Ne reçoit JAMAIS le texte brut du CV —
uniquement les objets structurés déjà validés, donc pas de surface d'attaque prompt-injection
à ce stade (le sandwiching n'est même pas nécessaire ici, l'entrée est déjà des données propres).
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

try:
    from mistralai.client import Mistral
except ImportError:
    from mistralai import Mistral

from prompts.system_prompts import EXPLANATION_SYSTEM_PROMPT

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")


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

    response = client.chat.complete(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=0.3,
        messages=[
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
    )
    explanation = response.choices[0].message.content
    if not isinstance(explanation, str):
        explanation = str(explanation)
    return {"explanation": explanation}
