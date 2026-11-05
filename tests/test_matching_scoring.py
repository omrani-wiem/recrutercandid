import unittest
from unittest.mock import call, patch

from nodes.matching_scoring import matching_scoring_node, routing_decision, REJECT_THRESHOLD
import nodes.matching_scoring as scoring
from nodes.guardrails import guardrails_node
from state import JobRequirements, ResumeProfile


class MatchingScoringTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(scoring, "_load_embedding_model", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_unrelated_resume_scores_low_and_is_rejected(self):
        """
        Un profil totalement hors sujet (médecin candidatant à un poste de dev) doit
        obtenir un score bas et être rejeté — mais pas forcément un 0.0 exact. Un score
        exactement 0.0 était auparavant forcé par une règle spéciale qui provoquait un
        bug ailleurs : un vrai candidat qualifié pouvait aussi tomber à 0.0/20 dès que
        l'offre ne citait aucune compétence technique concrète, alors que son expérience
        et les filtres obligatoires étaient corrects — un résultat contradictoire et
        trompeur pour le recruteur. Le calcul pondéré normal (sans règle spéciale)
        aboutit déjà naturellement à un score bas dans un vrai cas de mismatch total,
        ce qui suffit à déclencher le rejet automatique.
        """
        resume = ResumeProfile(
            full_name="Dr. Alice Martin",
            years_experience=10,
            skills=["surgery", "diagnosis", "anatomy"],
            education=["Medicine"],
            certifications=[],
        )
        job = JobRequirements(
            title="Senior Python Engineer",
            required_skills=["python", "sql", "data engineering"],
            nice_to_have_skills=["airflow"],
            min_years_experience=5,
            required_certifications=[],
            location="Paris",
        )

        result = matching_scoring_node({"resume": resume, "job": job})
        score = result["score"]

        self.assertEqual(score.similarity_method, "bag_of_words")
        flags = guardrails_node({"resume": resume, "score": score})["security_flags"]
        self.assertFalse(any(flag["severity"] == "blocked" for flag in flags))
        self.assertLess(score.final_score, REJECT_THRESHOLD)
        self.assertEqual(routing_decision({"score": score}), "reject")

    def test_similarity_uses_skills_and_not_education_or_identity_fields(self):
        resume = ResumeProfile(
            full_name="Candidate One",
            email="one@example.com",
            years_experience=4,
            skills=["Python", "SQL"],
            education=["Master, born 1988, woman, nationality: French"],
            raw_sections={"identity": "Age: 38; gender: woman; religion: none"},
        )
        job = JobRequirements(
            title="Data Engineer",
            required_skills=["Python", "SQL"],
            nice_to_have_skills=["Airflow"],
            min_years_experience=3,
        )

        with patch.object(scoring, "_semantic_similarity", return_value=(0.8, "sentence_transformers")) as similarity:
            matching_scoring_node({"resume": resume, "job": job})

        similarity.assert_called_once_with("Python SQL", "Python SQL Airflow")


if __name__ == "__main__":
    unittest.main()
