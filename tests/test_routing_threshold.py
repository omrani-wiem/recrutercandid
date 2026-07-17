import unittest

from nodes.matching_scoring import routing_decision
from state import JobRequirements, MatchScore, ResumeProfile


class RoutingThresholdTests(unittest.TestCase):
    def _make_state(self, final_score: float) -> dict:
        resume = ResumeProfile(
            full_name="Alice Durand",
            email="alice@example.com",
            years_experience=6,
            skills=["python", "sql"],
            education=["Master"],
            certifications=[],
        )
        job = JobRequirements(
            title="Data Engineer",
            required_skills=["python", "sql"],
            nice_to_have_skills=[],
            min_years_experience=3,
            required_certifications=[],
            location="Paris",
        )
        score = MatchScore(
            semantic_similarity=0.8,
            skill_overlap_ratio=0.8,
            experience_match=0.9,
            hard_filters_passed=True,
            final_score=final_score,
            failed_filters=[],
            matched_skills=["python"],
            missing_skills=[],
        )
        return {"resume": resume, "job": job, "score": score}

    def test_score_above_14_is_shortlisted(self):
        self.assertEqual(routing_decision(self._make_state(14.1)), "shortlist")

    def test_score_below_8_is_rejected(self):
        self.assertEqual(routing_decision(self._make_state(7.9)), "reject")

    def test_score_in_gray_zone_is_human_review(self):
        self.assertEqual(routing_decision(self._make_state(10.0)), "needs_human_review")

    def test_score_at_shortlist_threshold_is_human_review(self):
        self.assertEqual(routing_decision(self._make_state(14.0)), "needs_human_review")

    def test_score_at_reject_threshold_is_human_review(self):
        self.assertEqual(routing_decision(self._make_state(8.0)), "needs_human_review")


if __name__ == "__main__":
    unittest.main()
