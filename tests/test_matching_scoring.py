import unittest

from nodes.matching_scoring import matching_scoring_node, routing_decision, REJECT_THRESHOLD
from state import JobRequirements, ResumeProfile


class MatchingScoringTests(unittest.TestCase):
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

        self.assertLess(score.final_score, REJECT_THRESHOLD)
        self.assertEqual(routing_decision({"score": score}), "reject")


if __name__ == "__main__":
    unittest.main()
