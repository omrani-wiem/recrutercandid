import unittest

from nodes.matching_scoring import matching_scoring_node
from state import JobRequirements, ResumeProfile


class SkillRelevanceIntegrationTests(unittest.TestCase):
    """
    Vérifie que matching_scoring_node utilise bien le jugement du nœud skill_relevance
    (LLM) quand il est présent dans l'état, et retombe correctement sur la
    correspondance mot-à-mot classique quand il est absent ou invalide.
    """

    def setUp(self):
        self.resume = ResumeProfile(
            full_name="Jean Dupont",
            years_experience=6,
            skills=["Python", "SQL", "Airflow", "Azure", "AWS", "Spark"],
            education=[],
        )
        self.job = JobRequirements(
            title="Analyste de données",
            required_skills=["compétences en analyse de données", "connaissances de base en programmation"],
            nice_to_have_skills=[],
            min_years_experience=0,
        )

    def test_without_judgment_falls_back_to_keyword_matching(self):
        result = matching_scoring_node({"resume": self.resume, "job": self.job})
        score = result["score"]

        self.assertEqual(score.skill_matching_method, "keyword")
        # Aucune correspondance mot-à-mot possible ici : l'offre ne cite aucune techno.
        self.assertEqual(score.skill_overlap_ratio, 0.0)

    def test_with_valid_judgment_uses_llm_assisted_overlap(self):
        judgment = {
            "matched_required": [
                "compétences en analyse de données",
                "connaissances de base en programmation",
            ],
            "missing_required": [],
            "matched_nice_to_have": [],
            "matching_candidate_skills": ["Python", "SQL", "Spark"],
        }

        result = matching_scoring_node({
            "resume": self.resume,
            "job": self.job,
            "skill_relevance_judgment": judgment,
        })
        score = result["score"]

        self.assertEqual(score.skill_matching_method, "llm_assisted")
        self.assertEqual(score.skill_overlap_ratio, 1.0)
        self.assertIn("compétences en analyse de données", score.matched_skills)

    def test_with_malformed_judgment_falls_back_to_keyword_matching(self):
        malformed_judgment = {"matched_required": "pas une liste"}  # format invalide

        result = matching_scoring_node({
            "resume": self.resume,
            "job": self.job,
            "skill_relevance_judgment": malformed_judgment,
        })
        score = result["score"]

        self.assertEqual(score.skill_matching_method, "keyword")


if __name__ == "__main__":
    unittest.main()
