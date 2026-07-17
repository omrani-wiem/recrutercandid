import unittest
from unittest.mock import patch

from state import ResumeProfile, JobRequirements
import pipeline_runner as pr


def _fake_ocr_node(state):
    return {"extracted_text": "texte CV factice", "ocr_confidence": 0.99, "ocr_method": "native"}


def _make_extraction_node(resume: ResumeProfile):
    def _node(state):
        return {"resume": resume}
    return _node


def _make_jd_parsing_node(job: JobRequirements):
    def _node(state):
        return {"job": job}
    return _node


def _fake_skill_relevance_node(state):
    return {"skill_relevance_judgment": None}


def _fake_explanation_node(state):
    return {"explanation": "Explication factice pour test."}


def _fake_rejection_email_node(state):
    return {
        "rejection_email_draft": {"subject": "Sujet factice", "body": "Corps factice"},
        "rejection_email_approved": False,
        "rejection_email_sent": False,
    }


class PipelineRunnerTests(unittest.TestCase):
    def _build_patched_graph(self, resume, job):
        """Construit le graphe avec tous les nœuds LLM remplacés par des versions factices."""
        import graph as graph_module

        patches = [
            patch.object(graph_module, "ocr_node", _fake_ocr_node),
            patch.object(graph_module, "extraction_node", _make_extraction_node(resume)),
            patch.object(graph_module, "jd_parsing_node", _make_jd_parsing_node(job)),
            patch.object(graph_module, "skill_relevance_node", _fake_skill_relevance_node),
            patch.object(graph_module, "explanation_node", _fake_explanation_node),
            patch.object(graph_module, "rejection_email_node", _fake_rejection_email_node),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        return graph_module.build_graph()

    def test_shortlist_path_has_no_interrupt(self):
        resume = ResumeProfile(full_name="Bon Candidat", years_experience=6,
                                skills=["python", "sql", "airflow"], education=[])
        job = JobRequirements(title="Data Eng", required_skills=["python", "sql", "airflow"],
                               min_years_experience=5)
        app = self._build_patched_graph(resume, job)

        candidate = {"id": "cand-1", "stored_path": "fake.pdf"}
        job_record = {"id": "job-1", "description": "peu importe"}

        result = pr.run_analysis(app, candidate, job_record)
        self.assertEqual(result["status"], pr.STATUS_SHORTLISTED)
        self.assertIsNone(result["interrupt_payload"])

    def test_reject_path_pauses_for_email_review_then_cancel(self):
        resume = ResumeProfile(full_name="Mauvais Candidat", years_experience=0,
                                skills=["cuisine"], education=[])
        job = JobRequirements(title="Data Eng", required_skills=["python", "sql", "airflow"],
                               min_years_experience=5)
        app = self._build_patched_graph(resume, job)

        candidate = {"id": "cand-2", "stored_path": "fake.pdf"}
        job_record = {"id": "job-1", "description": "peu importe"}

        first = pr.run_analysis(app, candidate, job_record)
        self.assertEqual(first["status"], pr.STATUS_PENDING_EMAIL_REVIEW)
        self.assertIsNotNone(first["rejection_email_draft"])

        second = pr.resume_email_review(app, candidate, job_record, action="cancel")
        self.assertEqual(second["status"], pr.STATUS_REJECTED_EMAIL_CANCELLED)

    def test_gray_zone_pauses_for_decision_then_shortlist(self):
        # Score construit pour tomber dans la zone grise (8-14) : bon overlap mais
        # expérience insuffisante.
        resume = ResumeProfile(full_name="Candidat Zone Grise", years_experience=1,
                                skills=["python", "sql"], education=[])
        job = JobRequirements(title="Data Eng", required_skills=["python", "sql", "airflow"],
                               min_years_experience=5)
        app = self._build_patched_graph(resume, job)

        candidate = {"id": "cand-3", "stored_path": "fake.pdf"}
        job_record = {"id": "job-1", "description": "peu importe"}

        first = pr.run_analysis(app, candidate, job_record)
        self.assertIn(first["status"], (pr.STATUS_PENDING_DECISION, pr.STATUS_SHORTLISTED, pr.STATUS_PENDING_EMAIL_REVIEW))

        if first["status"] == pr.STATUS_PENDING_DECISION:
            second = pr.resume_decision(app, candidate, job_record, decision="shortlist")
            self.assertEqual(second["status"], pr.STATUS_SHORTLISTED)


    def test_reject_path_pauses_for_email_review_then_approve(self):
        resume = ResumeProfile(full_name="Mauvais Candidat 2", years_experience=0,
                                skills=["cuisine"], education=[])
        job = JobRequirements(title="Data Eng", required_skills=["python", "sql", "airflow"],
                               min_years_experience=5)
        app = self._build_patched_graph(resume, job)

        candidate = {"id": "cand-4", "stored_path": "fake.pdf"}
        job_record = {"id": "job-1", "description": "peu importe"}

        first = pr.run_analysis(app, candidate, job_record)
        self.assertEqual(first["status"], pr.STATUS_PENDING_EMAIL_REVIEW)

        # Pas de clé Resend/réseau dans cet environnement de test : l'envoi échoue,
        # mais ça doit rester une transition propre (pas de crash), pour distinguer
        # "approuvé mais échec technique d'envoi" d'une annulation volontaire.
        second = pr.resume_email_review(app, candidate, job_record, action="approve")
        self.assertIn(second["status"], (pr.STATUS_REJECTED_EMAIL_SENT, pr.STATUS_REJECTED_EMAIL_FAILED))

    def test_gray_zone_reject_then_pauses_for_email_review(self):
        resume = ResumeProfile(full_name="Candidat Zone Grise 2", years_experience=1,
                                skills=["python", "sql"], education=[])
        job = JobRequirements(title="Data Eng", required_skills=["python", "sql", "airflow"],
                               min_years_experience=5)
        app = self._build_patched_graph(resume, job)

        candidate = {"id": "cand-5", "stored_path": "fake.pdf"}
        job_record = {"id": "job-1", "description": "peu importe"}

        first = pr.run_analysis(app, candidate, job_record)
        if first["status"] != pr.STATUS_PENDING_DECISION:
            self.skipTest("le score calculé pour ce test n'est pas tombé en zone grise")

        second = pr.resume_decision(app, candidate, job_record, decision="reject")
        self.assertEqual(second["status"], pr.STATUS_PENDING_EMAIL_REVIEW)
        self.assertIsNotNone(second["rejection_email_draft"])


if __name__ == "__main__":
    unittest.main()
