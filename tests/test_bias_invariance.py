import unittest
from unittest.mock import patch

import nodes.matching_scoring as scoring
from nodes.matching_scoring import matching_scoring_node, routing_decision
from state import JobRequirements, ResumeProfile


class CounterfactualBiasInvarianceTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(scoring, "_load_embedding_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_personal_information_changes_do_not_change_scores_or_decisions(self):
        job = JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL", "Airflow"],
            nice_to_have_skills=["Spark"],
            min_years_experience=5,
        )
        candidate_profiles = [
            {
                "years_experience": 6,
                "skills": ["Python", "SQL", "Airflow", "Spark"],
                "certifications": [],
                "expected_route": "shortlist",
            },
            {
                "years_experience": 2,
                "skills": ["Python", "SQL"],
                "certifications": [],
                "expected_route": "needs_human_review",
            },
            {
                "years_experience": 0,
                "skills": ["patient care", "nursing", "triage"],
                "certifications": [],
                "expected_route": "reject",
            },
            {
                "years_experience": 7,
                "skills": ["Python", "SQL", "Airflow", "Spark"],
                "certifications": [],
                "expected_route": "reject",
            },
        ]
        candidate_jobs = [
            job,
            job,
            job,
            JobRequirements(
                title="Data Engineer",
                required_skills=["Python", "SQL", "Airflow", "Spark"],
                required_certifications=["AWS"],
                min_years_experience=5,
            ),
        ]
        identity_variants = [
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
                "raw_sections": {"identity": "Âge : 29 ans; situation familiale : marié"},
            },
            {
                "age_group": "over_40",
                "gender_group": "man",
                "full_name": "Alex Martin",
                "email": "alex.m@example.test",
                "education": ["Master informatique; born 1978; gender: man; nationality: French"],
                "raw_sections": {"identity": "Âge : 48 ans; religion : déclarée"},
            },
        ]

        group_routes: dict[tuple[str, str], dict[str, str]] = {}
        for case_index, candidate_case in enumerate(candidate_profiles):
            expected_score = None
            expected_route = None
            for identity in identity_variants:
                resume = ResumeProfile(
                    years_experience=candidate_case["years_experience"],
                    skills=candidate_case["skills"],
                    certifications=candidate_case["certifications"],
                    **{
                        key: value
                        for key, value in identity.items()
                        if key not in {"age_group", "gender_group"}
                    },
                )
                score = matching_scoring_node({
                    "resume": resume,
                    "job": candidate_jobs[case_index],
                })["score"]
                route = routing_decision({"score": score})
                for group_key in (
                    ("age_group", identity["age_group"]),
                    ("gender_group", identity["gender_group"]),
                ):
                    group_routes.setdefault(group_key, {})[str(case_index)] = route

                if expected_score is None:
                    expected_score = score.model_dump()
                    expected_route = route
                else:
                    self.assertEqual(score.model_dump(), expected_score)
                    self.assertEqual(route, expected_route)

            self.assertEqual(expected_route, candidate_case["expected_route"])

        expected_routes = {
            str(index): candidate_case["expected_route"]
            for index, candidate_case in enumerate(candidate_profiles)
        }
        for routes in group_routes.values():
            self.assertEqual(routes, expected_routes)


if __name__ == "__main__":
    unittest.main()
