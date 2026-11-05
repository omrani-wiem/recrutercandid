import unittest
from unittest.mock import patch

import numpy as np

import nodes.matching_scoring as scoring


class _FakeSentenceTransformer:
    def encode(self, texts, **kwargs):
        vectors = {
            "développement logiciel avec python": [1.0, 0.0],
            "création d'applications en python": [0.8, 0.6],
            "gestion des stocks et logistique": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


class _BrokenSentenceTransformer:
    def encode(self, texts, **kwargs):
        raise RuntimeError("model inference failed")


class SemanticSimilarityTests(unittest.TestCase):
    def test_sentence_transformer_similarity_distinguishes_paraphrase_and_mismatch(self):
        with patch.object(scoring, "_load_embedding_model", return_value=_FakeSentenceTransformer()):
            paraphrase_similarity, paraphrase_method = scoring._semantic_similarity(
                "développement logiciel avec python",
                "création d'applications en python",
            )
            unrelated_similarity, unrelated_method = scoring._semantic_similarity(
                "développement logiciel avec python",
                "gestion des stocks et logistique",
            )

        self.assertEqual(paraphrase_method, "sentence_transformers")
        self.assertEqual(unrelated_method, "sentence_transformers")
        self.assertGreater(paraphrase_similarity, unrelated_similarity)
        self.assertAlmostEqual(paraphrase_similarity, 0.8)
        self.assertEqual(unrelated_similarity, 0.0)

    def test_inference_error_uses_bag_of_words_fallback(self):
        previous_failure_state = scoring._embedding_runtime_failed
        scoring._embedding_runtime_failed = False
        self.addCleanup(
            setattr,
            scoring,
            "_embedding_runtime_failed",
            previous_failure_state,
        )

        with patch.object(scoring, "_load_embedding_model", return_value=_BrokenSentenceTransformer()):
            similarity, method = scoring._semantic_similarity("python sql", "python sql")

        self.assertEqual(method, "bag_of_words")
        self.assertAlmostEqual(similarity, 1.0)

    def test_bag_of_words_is_used_when_model_is_unavailable(self):
        with patch.object(scoring, "_load_embedding_model", return_value=None):
            similarity, method = scoring._semantic_similarity("python sql", "python sql")

        self.assertEqual(method, "bag_of_words")
        self.assertAlmostEqual(similarity, 1.0)

    def test_empty_text_returns_zero_without_loading_model(self):
        with patch.object(scoring, "_load_embedding_model") as load_model:
            similarity, method = scoring._semantic_similarity("", "python sql")

        load_model.assert_not_called()
        self.assertEqual(similarity, 0.0)
        self.assertEqual(method, "bag_of_words")


if __name__ == "__main__":
    unittest.main()
