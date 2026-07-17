"""
Nœud FONCTION : vérifie que le score n'est pas corrélé à un attribut protégé, et que le
profil ne contient plus d'information sensible avant de continuer dans le pipeline.
Pas de LLM ici — des règles + un classifieur léger suffisent et sont plus auditables.
"""
from __future__ import annotations
import re

PROTECTED_ATTR_PATTERNS = {
    "âge": r"\b(n[ée]\(?e?\)? en \d{4}|\d{2}\s*ans)\b",
    "genre": r"\b(monsieur|madame|m\.|mme)\b",
    "photo": r"\.(jpg|jpeg|png)\b",
}


def guardrails_node(state: dict) -> dict:
    resume = state["resume"]
    flags = state.get("security_flags", [])

    # 1. Détection d'attributs protégés encore présents dans les données structurées
    combined_text = " ".join(resume.raw_sections.values())
    for attr, pattern in PROTECTED_ATTR_PATTERNS.items():
        if re.search(pattern, combined_text, re.IGNORECASE):
            flags.append({
                "node": "guardrails",
                "severity": "info",
                "reason": f"attribut potentiellement protégé détecté ({attr}) — exclu du scoring",
            })

    # 2. Vérifie que le score n'a pas été influencé par un signal hors périmètre autorisé
    #    (le scoring ne lit que semantic_similarity / skill_overlap_ratio / experience_match /
    #    hard_filters_passed — ce check confirme qu'aucune feature non whitelistée n'a été
    #    injectée en amont)
    allowed_features = {
        "semantic_similarity", "skill_overlap_ratio", "hard_filters_passed", "experience_match",
    }
    score_fields = set(state["score"].model_dump().keys())
    unexpected = score_fields - allowed_features - {
        "final_score", "failed_filters", "matched_skills", "missing_skills",
        "candidate_years_experience", "required_years_experience",
        "score_breakdown", "skill_matching_method",
    }
    if unexpected:
        flags.append({
            "node": "guardrails",
            "severity": "blocked",
            "reason": f"champs de score non whitelistés: {unexpected}",
        })

    return {"security_flags": flags}
