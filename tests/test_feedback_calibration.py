import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feedback_calibration import feedback_store, scoring_model


class FeedbackStoreMigrationTests(unittest.TestCase):
    def test_reads_and_clears_legacy_and_current_feedback_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_log = root / "rl" / "feedback_log.jsonl"
            current_log = root / "feedback_calibration" / "feedback_log.jsonl"
            legacy_log.parent.mkdir()
            current_log.parent.mkdir()
            legacy_log.write_text('{"reward": 0}\n', encoding="utf-8")
            current_log.write_text('{"feedback_label": 1}\n', encoding="utf-8")

            with (
                patch.object(feedback_store, "LEGACY_FEEDBACK_LOG_PATH", legacy_log),
                patch.object(feedback_store, "FEEDBACK_LOG_PATH", current_log),
            ):
                entries = feedback_store.load_feedback_log()
                feedback_store.clear_feedback_log()

            self.assertEqual(
                [entry.get("feedback_label", entry.get("reward")) for entry in entries],
                [0, 1],
            )
            self.assertFalse(legacy_log.exists())
            self.assertFalse(current_log.exists())


class ScoringWeightMigrationTests(unittest.TestCase):
    def test_loads_legacy_weights_and_saves_calibrated_weights_to_new_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy_weights = root / "rl" / "weights.json"
            current_weights = root / "feedback_calibration" / "weights.json"
            legacy_weights.parent.mkdir()
            current_weights.parent.mkdir()
            legacy_weights.write_text(
                json.dumps({"bias": 0.25}),
                encoding="utf-8",
            )

            with (
                patch.object(scoring_model, "LEGACY_MODEL_PATH", legacy_weights),
                patch.object(scoring_model, "MODEL_PATH", current_weights),
            ):
                loaded_model = scoring_model.ScoringModel.load_latest()
                self.assertEqual(loaded_model.weights["bias"], 0.25)
                updated_model = scoring_model.calibrate_scoring_weights([
                    {
                        "features": {
                            "semantic_similarity": 0.9,
                            "skill_overlap_ratio": 0.8,
                            "experience_match": 1.0,
                            "hard_filters_passed": 1.0,
                        },
                        "feedback_label": 1.0,
                    },
                    {
                        "features": {
                            "semantic_similarity": 0.1,
                            "skill_overlap_ratio": 0.0,
                            "experience_match": 0.2,
                            "hard_filters_passed": 0.0,
                        },
                        "feedback_label": 0.0,
                    },
                ])

            self.assertTrue(current_weights.exists())
            self.assertEqual(
                json.loads(current_weights.read_text(encoding="utf-8")),
                updated_model.weights,
            )
            self.assertTrue(legacy_weights.exists())


if __name__ == "__main__":
    unittest.main()
