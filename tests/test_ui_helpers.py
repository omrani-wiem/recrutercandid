import unittest

from ui_helpers import candidates_to_analyze


class CandidatesToAnalyzeTests(unittest.TestCase):
    def test_retries_missing_and_error_results_only(self):
        candidates = [
            {"results": {"job1": {"status": "error"}}},
            {"results": {"job1": {"status": "shortlisted"}}},
            {"results": {}},
        ]

        pending = candidates_to_analyze(candidates, "job1")

        self.assertEqual(len(pending), 2)
        self.assertEqual(pending[0]["results"]["job1"]["status"], "error")
        self.assertEqual(pending[1]["results"], {})


if __name__ == "__main__":
    unittest.main()
