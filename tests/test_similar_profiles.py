import unittest
from unittest.mock import patch

import numpy as np

import nodes.matching_scoring as scoring


class _FakeEmbeddingModel:
    def encode(self, texts, **kwargs):
        vectors = {
            "python": [1.0, 0.0],
            "python sql": [1.0, 0.0],
            "python data": [0.9, 0.1],
            "legal writing": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


class SimilarProfileRankingTests(unittest.TestCase):
    def setUp(self):
        previous_failure_state = scoring._embedding_runtime_failed
        self.addCleanup(
            setattr,
            scoring,
            "_embedding_runtime_failed",
            previous_failure_state,
        )

    def test_ranks_profiles_by_sentence_embedding_similarity(self):
        profiles = [
            ("unrelated", ["legal writing"]),
            ("similar", ["python data"]),
        ]
        with patch.object(
            scoring, "_load_embedding_model", return_value=_FakeEmbeddingModel()
        ):
            ranked, method = scoring.rank_similar_profiles(["python", "sql"], profiles)

        self.assertEqual(method, "sentence_transformers")
        self.assertEqual([candidate_id for candidate_id, _ in ranked], ["similar", "unrelated"])
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_uses_bag_of_words_when_embedding_model_is_unavailable(self):
        profiles = [
            ("partial-match", ["python"]),
            ("same-skills", ["python", "sql"]),
        ]
        with patch.object(scoring, "_load_embedding_model", return_value=None):
            ranked, method = scoring.rank_similar_profiles(["python", "sql"], profiles)

        self.assertEqual(method, "bag_of_words")
        self.assertEqual(ranked[0][0], "same-skills")
        self.assertAlmostEqual(ranked[0][1], 1.0)

    def test_empty_skills_get_zero_similarity(self):
        with patch.object(
            scoring, "_load_embedding_model", return_value=_FakeEmbeddingModel()
        ):
            ranked, _ = scoring.rank_similar_profiles(
                ["python"], [("empty", []), ("other", ["legal writing"])]
            )

        self.assertEqual(dict(ranked)["empty"], 0.0)


if __name__ == "__main__":
    unittest.main()
