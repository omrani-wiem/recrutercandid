from __future__ import annotations

import re
from collections import Counter
import numpy as np

from state import MatchScore
from rl.scoring_model import ScoringModel

_scoring_model = ScoringModel.load_latest()

# Seuils de routage, exprimés sur l'échelle finale /20 (cf. routing_decision).
# Un score STRICTEMENT au-dessus de SHORTLIST_THRESHOLD passe en shortlist direct.
# Un score STRICTEMENT en dessous de REJECT_THRESHOLD est rejeté direct.
# Entre les deux : zone grise -> revue humaine (human_review_decision).
REJECT_THRESHOLD = 8.0
SHORTLIST_THRESHOLD = 14.0


def _embed(texts: list[str]) -> np.ndarray:
    """
    Embeddings locaux très simples basés sur un bag-of-words.
    """
    tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for text in texts]
    vocab = sorted({token for tokens in tokenized for token in tokens})

    vectors = []
    for tokens in tokenized:
        counts = Counter(tokens)
        vector = np.array([counts.get(token, 0) for token in vocab], dtype=float)
        vectors.append(vector)

    return np.array(vectors)


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Similarité cosinus normalisée sur [0, 1].
    """
    embeddings = _embed([text_a, text_b])
    vec_a, vec_b = embeddings[0], embeddings[1]

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, sim))


def _skill_overlap_from_judgment(
    judgment: dict,
    required: list[str],
) -> tuple[float, list[str], list[str]] | None:
    """
    Calcule l'overlap à partir du jugement du nœud skill_relevance (LLM), quand
    disponible et bien formé. Retourne None si le jugement est absent/invalide, pour
    que l'appelant retombe sur la correspondance mot-à-mot classique (_skill_overlap).
    """
    if not isinstance(judgment, dict):
        return None

    matched_required = judgment.get("matched_required")
    missing_required = judgment.get("missing_required")
    if not isinstance(matched_required, list) or not isinstance(missing_required, list):
        return None

    required_count = len(required or [])
    if required_count == 0:
        return 0.0, [], []

    overlap_ratio = max(0.0, min(1.0, len(matched_required) / required_count))
    return overlap_ratio, sorted(str(s) for s in matched_required), sorted(str(s) for s in missing_required)


def _skill_overlap(
    resume_skills: list[str],
    required: list[str],
    nice_to_have: list[str],
) -> tuple[float, list[str], list[str]]:
    """
    Overlap via une taxonomie simple de synonymes plutôt qu'un match exact.
    Le ratio principal reste basé sur les skills requises.

    C'est la méthode de repli (skill_matching_method="keyword") : purement lexicale,
    donc aveugle aux cas où l'offre décrit une compétence en termes généraux (ex:
    "compétences en analyse de données") plutôt que par son nom concret ("Python").
    Voir nodes/skill_relevance.py pour la méthode assistée par LLM, utilisée en priorité
    quand disponible (voir matching_scoring_node).
    """
    synonyms = {
        "js": "javascript",
        "ts": "typescript",
        "k8s": "kubernetes",
        "py": "python",
        "postgres": "postgresql",
        "postgresql": "postgresql",
    }

    def normalize(s: str) -> str:
        return synonyms.get(s.lower().strip(), s.lower().strip())

    resume_norm = {normalize(s) for s in (resume_skills or [])}
    required_norm = {normalize(s) for s in (required or [])}
    nice_norm = {normalize(s) for s in (nice_to_have or [])}

    matched = resume_norm & (required_norm | nice_norm)
    missing_required = required_norm - resume_norm

    overlap_ratio = len(resume_norm & required_norm) / max(len(required_norm), 1)
    overlap_ratio = max(0.0, min(1.0, overlap_ratio))

    return overlap_ratio, sorted(matched), sorted(missing_required)


def _experience_match(
    candidate_years: float | int | None,
    required_min_years: float | int | None
) -> float:
    """
    Score continu entre 0 et 1 pour l'expérience.

    Règle métier :
    - si le poste demande MINIMUM 5 ans, alors 5, 6 ou 8 ans doivent donner un score élevé
    - si le candidat est légèrement en dessous (ex: 4 pour 5), on évite un score nul
    - si le candidat est très loin du minimum, le score baisse
    """
    if required_min_years is None or required_min_years <= 0:
        return 1.0

    if candidate_years is None:
        return 0.0

    candidate_years = float(candidate_years)
    required_min_years = float(required_min_years)

    if candidate_years < 0:
        return 0.0

    # si le minimum est atteint ou dépassé -> score max
    if candidate_years >= required_min_years:
        return 1.0

    # en dessous du minimum -> pénalité progressive douce
    ratio = candidate_years / required_min_years
    return max(0.0, min(1.0, ratio ** 0.5))


def _hard_filters(resume, job) -> tuple[bool, list[str]]:
    """
    Filtres vraiment bloquants.
    IMPORTANT : on ne met PAS l'expérience ici, sinon on retombe dans un rejet binaire.
    """
    failed = []

    resume_certs = set(getattr(resume, "certifications", []) or [])
    required_certs = set(getattr(job, "required_certifications", []) or [])

    missing_certs = required_certs - resume_certs
    if missing_certs:
        failed.append(f"certifications manquantes: {sorted(missing_certs)}")

    return len(failed) == 0, failed


def _compute_score_breakdown(
    semantic_similarity: float,
    overlap_ratio: float,
    experience_match: float,
    filters_ok: bool,
) -> dict[str, float]:
    """
    Prépare un breakdown exploitable par l'UI pour expliquer le score.
    """
    weights = _scoring_model.weights
    filters_value = float(filters_ok)

    return {
        "semantic_similarity": round(semantic_similarity, 3),
        "skill_overlap_ratio": round(overlap_ratio, 3),
        "experience_match": round(experience_match, 3),
        "hard_filters_passed": filters_value,

        "contrib_semantic": round(weights.get("semantic_similarity", 0.0) * semantic_similarity, 3),
        "contrib_skills": round(weights.get("skill_overlap_ratio", 0.0) * overlap_ratio, 3),
        "contrib_experience": round(weights.get("experience_match", 0.0) * experience_match, 3),
        "contrib_filters": round(weights.get("hard_filters_passed", 0.0) * filters_value, 3),
        "bias": round(weights.get("bias", 0.0), 3),
    }


def matching_scoring_node(state: dict) -> dict:
    resume, job = state["resume"], state["job"]

    resume_skills = getattr(resume, "skills", []) or []
    resume_education = getattr(resume, "education", []) or []
    job_required_skills = getattr(job, "required_skills", []) or []
    job_nice_skills = getattr(job, "nice_to_have_skills", []) or []

    resume_text = " ".join(resume_skills + resume_education)
    job_text = " ".join(job_required_skills + job_nice_skills)

    semantic_similarity = _semantic_similarity(resume_text, job_text)

    overlap_from_judgment = _skill_overlap_from_judgment(
        state.get("skill_relevance_judgment"),
        job_required_skills,
    )
    if overlap_from_judgment is not None:
        overlap_ratio, matched_skills, missing_skills = overlap_from_judgment
        skill_matching_method = "llm_assisted"
    else:
        overlap_ratio, matched_skills, missing_skills = _skill_overlap(
            resume_skills,
            job_required_skills,
            job_nice_skills,
        )
        skill_matching_method = "keyword"

    candidate_years = float(getattr(resume, "years_experience", 0) or 0)
    required_years = float(getattr(job, "min_years_experience", 0) or 0)

    experience_match = _experience_match(candidate_years, required_years)
    filters_ok, failed_filters = _hard_filters(resume, job)

    # NB : il y avait ici une règle qui forçait final_score à exactement 0.0 dès que
    # overlap_ratio ET semantic_similarity étaient très bas. Elle a été retirée : elle
    # provoquait un score de 0.0/20 affiché à côté de "Expérience : Suffisante" et
    # "Conditions obligatoires : Respectées" — contradictoire et trompeur — dans les cas
    # (fréquents) où l'offre ne cite aucune compétence technique concrète (ex: "compétences
    # en analyse de données" au lieu de "Python, SQL"). Dans ce cas, l'absence de
    # correspondance mot-à-mot ne veut PAS dire que le candidat est hors-sujet : ça veut
    # dire que l'offre est formulée de façon trop générale pour être comparée mot-à-mot.
    # Le calcul pondéré normal ci-dessous suffit : un vrai mismatch total (compétences ET
    # similarité ET expérience faibles) obtient déjà naturellement un score bas qui route
    # vers le rejet automatique, sans avoir besoin d'un score contradictoire à 0.0 pile.
    features = {
        "semantic_similarity": semantic_similarity,
        "skill_overlap_ratio": overlap_ratio,
        "experience_match": experience_match,
        "hard_filters_passed": float(filters_ok),
    }

    raw_score = _scoring_model.predict(features)
    final_score = max(0.0, min(1.0, raw_score))

    # NB: on a aussi volontairement retiré les anciens ajustements qui forçaient le score
    # à une valeur FIXE (ex: `final_score = max(final_score, 0.70)`). Ce clamp
    # écrasait la différenciation entre candidats : dès que l'expérience était
    # couverte et l'overlap de compétences >= 50%, tout le monde retombait
    # exactement sur 0.70 * 20 = 14/20, quel que soit le reste du profil.
    # Le modèle pondéré (_scoring_model.predict) encode déjà l'importance de
    # l'expérience et des compétences via ses poids ; on n'a pas besoin d'un
    # second mécanisme qui le contredit en écrasant sa sortie.

    score_breakdown = _compute_score_breakdown(
        semantic_similarity=semantic_similarity,
        overlap_ratio=overlap_ratio,
        experience_match=experience_match,
        filters_ok=filters_ok,
    )

    final_score_out_of_20 = round(final_score * 20, 1)

    score = MatchScore(
        semantic_similarity=round(semantic_similarity, 3),
        skill_overlap_ratio=round(overlap_ratio, 3),
        experience_match=round(experience_match, 3),
        hard_filters_passed=filters_ok,
        final_score=final_score_out_of_20,
        candidate_years_experience=candidate_years,
        required_years_experience=required_years,
        failed_filters=failed_filters,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        skill_matching_method=skill_matching_method,
        score_breakdown=score_breakdown,
    )

    return {"score": score}


def routing_decision(state: dict) -> str:
    """
    Branchement conditionnel post-scoring.

    - score > SHORTLIST_THRESHOLD             -> shortlist direct
    - score < REJECT_THRESHOLD                -> reject direct
    - REJECT_THRESHOLD <= score <= SHORTLIST_THRESHOLD -> zone grise, revue humaine

    (Avant correctif, le seuil bas et le seuil haut étaient tous les deux fixés à 13.0,
    ce qui rendait la zone grise quasiment inatteignable : avec un score continu, il
    fallait tomber EXACTEMENT sur 13.0 pour déclencher une revue humaine.)
    """
    score = state["score"]

    if not score.hard_filters_passed:
        return "reject"

    if score.final_score > SHORTLIST_THRESHOLD:
        return "shortlist"

    if score.final_score < REJECT_THRESHOLD:
        return "reject"

    return "needs_human_review"
