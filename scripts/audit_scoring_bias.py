"""Run a synthetic paired counterfactual audit of score invariance to identity data."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes.matching_scoring import matching_scoring_node, routing_decision
from state import JobRequirements, ResumeProfile


IDENTITY_VARIANTS = [
    {
        "age_group": "under_35",
        "gender_group": "woman",
        "full_name": "Alex Martin",
        "email": "alex@example.test",
        "education": ["Master informatique; born 1997; gender: woman; nationality: French"],
        "raw_sections": {"identity": "Age: 29; marital status: single"},
    },
    {
        "age_group": "over_40",
        "gender_group": "woman",
        "full_name": "Samira Haddad",
        "email": "samira@example.test",
        "education": ["Master informatique; born 1978; gender: woman; nationality: Tunisian"],
        "raw_sections": {"identity": "Age: 48; marital status: married"},
    },
    {
        "age_group": "under_35",
        "gender_group": "man",
        "full_name": "Samir Haddad",
        "email": "samir@example.test",
        "education": ["Master informatique; born 1997; gender: man; nationality: Tunisian"],
        "raw_sections": {"identity": "Age: 29; marital status: married"},
    },
    {
        "age_group": "over_40",
        "gender_group": "man",
        "full_name": "Alex Martin",
        "email": "alex.m@example.test",
        "education": ["Master informatique; born 1978; gender: man; nationality: French"],
        "raw_sections": {"identity": "Age: 48; religion: disclosed"},
    },
]

COUNTERFACTUAL_CASES = [
    {
        "label": "strong_match",
        "years_experience": 7,
        "skills": ["Python", "SQL", "Airflow", "Spark"],
        "job": JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow", "Spark"],
            min_years_experience=5,
        ),
    },
    {
        "label": "partial_match",
        "years_experience": 2,
        "skills": ["Python", "SQL"],
        "job": JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow"],
            min_years_experience=5,
        ),
    },
    {
        "label": "unrelated_profile",
        "years_experience": 0,
        "skills": ["nursing", "patient care", "triage"],
        "job": JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow"],
            min_years_experience=5,
        ),
    },
    {
        "label": "missing_required_certification",
        "years_experience": 7,
        "skills": ["Python", "SQL", "Airflow", "Spark"],
        "job": JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow", "Spark"],
            required_certifications=["AWS"],
            min_years_experience=5,
        ),
    },
]


def main() -> None:
    group_routes: dict[tuple[str, str], dict[str, str]] = {}
    pair_count = 0

    for case in COUNTERFACTUAL_CASES:
        reference_score = None
        reference_route = None

        for identity in IDENTITY_VARIANTS:
            resume = ResumeProfile(
                years_experience=case["years_experience"],
                skills=case["skills"],
                certifications=[],
                **{
                    key: value
                    for key, value in identity.items()
                    if key not in {"age_group", "gender_group"}
                },
            )
            score = matching_scoring_node({"resume": resume, "job": case["job"]})["score"]
            route = routing_decision({"score": score})
            pair_count += 1

            if reference_score is None:
                reference_score = score.model_dump()
                reference_route = route
            elif score.model_dump() != reference_score or route != reference_route:
                raise AssertionError(
                    f"Counterfactual mismatch for {case['label']} with "
                    f"{identity['age_group']}/{identity['gender_group']}: "
                    f"{score.final_score}/20, {route}; expected "
                    f"{reference_score['final_score']}/20, {reference_route}"
                )

            for group_key in (
                ("age_group", identity["age_group"]),
                ("gender_group", identity["gender_group"]),
            ):
                group_routes.setdefault(group_key, {})[case["label"]] = route

        print(
            f"{case['label']}: score={reference_score['final_score']}/20, "
            f"decision={reference_route} (identical across all identity variants)"
        )

    print(f"\nCounterfactual pairs checked: {pair_count}")
    print("Synthetic shortlist rates by demographic label (not population statistics):")
    for (dimension, label), case_routes in sorted(group_routes.items()):
        routes = list(case_routes.values())
        counts = Counter(routes)
        shortlist_rate = counts["shortlist"] / len(routes)
        print(
            f"  {dimension}={label}: shortlist={shortlist_rate:.0%}; "
            f"review={counts['needs_human_review']}; reject={counts['reject']}"
        )
    print(
        "\nInterpretation: this checks paired invariance only. It is not a statistical "
        "fairness audit and does not test proxy effects through experience or skills."
    )


if __name__ == "__main__":
    main()
