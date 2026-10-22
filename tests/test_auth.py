import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import auth


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        data_dir = Path(self.tmp.name)
        patches = [
            mock.patch.object(auth, "DATA_DIR", data_dir),
            mock.patch.object(auth, "USERS_PATH", data_dir / "users.json"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_default_accounts_use_env_passwords(self):
        env = {"DEMO_RECRUITER_PASSWORD": "r-secret-1", "DEMO_STUDENT_PASSWORD": "e-secret-1"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(auth.verify_login("recruteur", "r-secret-1"), auth.ROLE_RECRUTEUR)
            self.assertEqual(auth.verify_login("etudiant", "e-secret-1"), auth.ROLE_ETUDIANT)

    def test_old_hardcoded_passwords_no_longer_work(self):
        with mock.patch.dict(os.environ, {"DEMO_RECRUITER_PASSWORD": "", "DEMO_STUDENT_PASSWORD": ""}):
            self.assertIsNone(auth.verify_login("recruteur", "recruteur"))
            self.assertIsNone(auth.verify_login("etudiant", "etudiant"))

    def test_wrong_password_and_unknown_user_are_rejected(self):
        with mock.patch.dict(os.environ, {"DEMO_RECRUITER_PASSWORD": "r-secret-1"}):
            self.assertIsNone(auth.verify_login("recruteur", "mauvais"))
            self.assertIsNone(auth.verify_login("inconnu", "r-secret-1"))

    def test_add_user_refuses_duplicates(self):
        self.assertTrue(auth.add_user("alice", "pw-alice-1", auth.ROLE_ETUDIANT))
        self.assertFalse(auth.add_user("alice", "autre", auth.ROLE_ETUDIANT))
        self.assertEqual(auth.verify_login("alice", "pw-alice-1"), auth.ROLE_ETUDIANT)


if __name__ == "__main__":
    unittest.main()
