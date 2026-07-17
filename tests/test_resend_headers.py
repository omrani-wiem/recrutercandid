import unittest

from graph import _build_resend_headers


class ResendHeadersTests(unittest.TestCase):
    def test_builds_user_agent_for_resend_requests(self):
        headers = _build_resend_headers("test-key")

        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["User-Agent"], "cv-matching-system/1.0")


if __name__ == "__main__":
    unittest.main()
