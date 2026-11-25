import tempfile, unittest
from pathlib import Path
from rl import scoring_model as sm

F = {"semantic_similarity": 0.5, "skill_overlap_ratio": 0.8,
     "experience_match": 1.0, "hard_filters_passed": 1.0}

class TestRecalibration(unittest.TestCase):
    def setUp(self):
        self._orig = sm.MODEL_PATH
        sm.MODEL_PATH = Path(tempfile.mkdtemp()) / "weights.json"

    def tearDown(self):
        sm.MODEL_PATH = self._orig

    def test_idempotent(self):
        log = [{"features": F, "reward": 1.0, "candidate_name": "A", "job_title": "X"}]
        w1 = sm.offline_update(log).weights
        w2 = sm.offline_update(log).weights
        self.assertEqual(w1, w2)

    def test_weights_stay_valid(self):
        log = [{"features": F, "reward": 0.0, "candidate_name": f"C{i}", "job_title": "X"}
               for i in range(50)]
        w = sm.offline_update(log).weights
        self.assertTrue(all(w[k] >= 0 for k in sm.FEATURE_ORDER))
        self.assertAlmostEqual(sum(w[k] for k in sm.FEATURE_ORDER), 1.0, places=6)

    def test_positive_feedback_raises_score(self):
        before = sm.ScoringModel().predict(F)
        log = [{"features": F, "reward": 1.0, "candidate_name": "A", "job_title": "X"}]
        after = sm.offline_update(log).predict(F)
        self.assertGreaterEqual(after, before)

if __name__ == "__main__":
    unittest.main()