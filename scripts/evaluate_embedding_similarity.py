"""Evaluate semantic similarities and score routing on synthetic French CV/job pairs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.matching_scoring import (
    _semantic_similarity,
    matching_scoring_node,
    routing_decision,
)
from state import JobRequirements, ResumeProfile


SIMILARITY_PAIRS = [
    (
        "data engineering",
        "Conception de pipelines de données, automatisation ETL et traitement distribué avec Spark.",
        "Ingénieur data : orchestration des flux, intégration et traitement de données avec Spark.",
    ),
    (
        "data analysis",
        "Analyse de données avec SQL, création de tableaux de bord et reporting.",
        "Interrogation SQL, visualisation de données, indicateurs et rapports de gestion.",
    ),
    (
        "finance",
        "Modélisation financière, reporting comptable et Excel avancé.",
        "Analyste financier, maîtrise d'Excel, prévisions budgétaires et reporting.",
    ),
    (
        "unrelated",
        "Infirmière hospitalière, soins aux patients et gestion des urgences.",
        "Développeur logiciel, architecture microservices Java et Kubernetes.",
    ),
]

ROUTING_CASES = [
    (
        "strong_match",
        ResumeProfile(
            full_name="Candidate A",
            years_experience=7,
            skills=["Python", "SQL", "Airflow", "Spark"],
            education=["Computer science"],
        ),
        JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow", "Spark"],
            min_years_experience=5,
        ),
        "shortlist",
    ),
    (
        "partial_match",
        ResumeProfile(
            full_name="Candidate B",
            years_experience=2,
            skills=["Python", "SQL"],
            education=["Information technology"],
        ),
        JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow"],
            min_years_experience=5,
        ),
        "needs_human_review",
    ),
    (
        "unrelated_profile",
        ResumeProfile(
            full_name="Candidate C",
            years_experience=0,
            skills=["nursing", "patient care", "triage"],
            education=["Nursing"],
        ),
        JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow"],
            min_years_experience=5,
        ),
        "reject",
    ),
]


def main() -> None:
    similarities = {}
    for label, candidate_text, job_text in SIMILARITY_PAIRS:
        similarity, method = _semantic_similarity(candidate_text, job_text)
        if method != "sentence_transformers":
            raise RuntimeError(
                f"{label}: expected Sentence Transformers, got fallback {method!r}"
            )
        similarities[label] = similarity
        print(f"similarity {label:16} {similarity:.3f}")

    related = [score for label, score in similarities.items() if label != "unrelated"]
    if min(related) <= similarities["unrelated"]:
        raise AssertionError("A related synthetic CV/job pair scored below the unrelated pair.")

    for label, resume, job, expected_route in ROUTING_CASES:
        score = matching_scoring_node({"resume": resume, "job": job})["score"]
        route = routing_decision({"score": score})
        print(f"routing    {label:16} {score.final_score:4.1f}/20 -> {route}")
        if route != expected_route:
            raise AssertionError(
                f"{label}: expected {expected_route}, received {route} "
                f"at {score.final_score}/20"
            )


if __name__ == "__main__":
    main()
