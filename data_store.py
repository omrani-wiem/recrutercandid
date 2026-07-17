from __future__ import annotations
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
JOBS_PATH = DATA_DIR / "jobs.json"
CANDIDATES_PATH = DATA_DIR / "candidates.json"
CV_FILES_DIR = DATA_DIR / "cv_files"

from prompts.domain_prompts import DOMAIN_SKILL_CONTEXT, DOMAIN_LABELS  # noqa: F401 (ré-exporté pour cohérence)

DOMAINS = list(DOMAIN_LABELS.keys())

def delete_result_for_candidate(candidate_id: str, job_id: str) -> None:
    """Supprime le résultat d'analyse d'un candidat pour une offre donnée."""
    candidates = load_candidates()
    for cand in candidates:
        if cand["id"] == candidate_id and job_id in cand.get("results", {}):
            del cand["results"][job_id]
    save_candidates(candidates)


def delete_candidate(candidate_id: str) -> None:
    """Supprime complètement un candidat : son fichier CV et son entrée."""
    candidates = load_candidates()
    remaining = []
    for c in candidates:
        if c["id"] == candidate_id:
            stored_path = Path(c["stored_path"])
            if stored_path.exists():
                stored_path.unlink()
        else:
            remaining.append(c)
    save_candidates(remaining)
    

def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CV_FILES_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: list) -> list:
    _ensure_dirs()
    if not path.exists():
        return list(default)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: list) -> None:
    _ensure_dirs()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Offres d'emploi
# ---------------------------------------------------------------------------

def _seed_jobs() -> list[dict]:
    """Quelques offres d'exemple par domaine, pour que l'interface ne soit pas vide
    au premier lancement. Modifiable ensuite directement dans data/jobs.json ou via
    l'interface."""
    return [
        {
            "id": "job-tech-1",
            "domain": "tech",
            "title": "Data Engineer Senior",
            "description": (
                "Nous recherchons un Data Engineer Senior pour concevoir et maintenir nos "
                "pipelines de données. Compétences requises : Python, SQL, Airflow. "
                "Apprécié : Spark, AWS. Minimum 5 ans d'expérience. Poste basé à Paris."
            ),
            "location": "Paris",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "job-finance-1",
            "domain": "finance",
            "title": "Analyste Financier",
            "description": (
                "Poste d'analyste financier au sein de la direction financière. Compétences "
                "requises : analyse de données financières, modélisation Excel avancée, "
                "reporting. Minimum 2 ans d'expérience. Poste basé à Lyon."
            ),
            "location": "Lyon",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "job-commerce-1",
            "domain": "commerce",
            "title": "Chargé(e) de Développement Commercial",
            "description": (
                "Rejoignez notre équipe commerciale pour développer notre portefeuille "
                "clients. Compétences requises : prospection, négociation, CRM. Minimum "
                "1 an d'expérience en vente B2B. Poste basé à Bordeaux."
            ),
            "location": "Bordeaux",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "id": "job-operations-1",
            "domain": "operations",
            "title": "Responsable Supply Chain",
            "description": (
                "Pilotage de la chaîne logistique et optimisation des flux. Compétences "
                "requises : gestion des stocks, planification, amélioration continue "
                "(Lean/Six Sigma). Minimum 3 ans d'expérience. Poste basé à Lille."
            ),
            "location": "Lille",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]


def load_jobs() -> list[dict]:
    jobs = _load_json(JOBS_PATH, [])
    if not jobs:
        jobs = _seed_jobs()
        _save_json(JOBS_PATH, jobs)
    return jobs


def get_job(job_id: str) -> dict | None:
    for job in load_jobs():
        if job["id"] == job_id:
            return job
    return None


def jobs_for_domain(domain: str) -> list[dict]:
    return [j for j in load_jobs() if j["domain"] == domain]


def add_job(domain: str, title: str, description: str, location: str = "") -> dict:
    jobs = load_jobs()
    job = {
        "id": f"job-{uuid.uuid4()}",
        "domain": domain,
        "title": title,
        "description": description,
        "location": location,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    jobs.append(job)
    _save_json(JOBS_PATH, jobs)
    return job


# ---------------------------------------------------------------------------
# Candidats (CV uploadés) et leurs candidatures
# ---------------------------------------------------------------------------

def load_candidates() -> list[dict]:
    return _load_json(CANDIDATES_PATH, [])


def save_candidates(candidates: list[dict]) -> None:
    _save_json(CANDIDATES_PATH, candidates)


def add_candidate_cv(file_bytes: bytes, original_filename: str, apply_to_all_jobs: bool = True) -> dict:
    """
    Enregistre un CV uploadé : copie le fichier dans data/cv_files/ et crée une entrée
    candidat. Par défaut, le candidat "postule" à toutes les offres existantes (comme
    demandé), pour simuler un vivier déjà candidaté sans avoir à répéter l'upload par offre.
    """
    _ensure_dirs()
    candidate_id = f"cand-{uuid.uuid4()}"
    stored_filename = f"{candidate_id}_{original_filename}"
    stored_path = CV_FILES_DIR / stored_filename
    stored_path.write_bytes(file_bytes)

    applied_job_ids = [j["id"] for j in load_jobs()] if apply_to_all_jobs else []

    candidate = {
        "id": candidate_id,
        "original_filename": original_filename,
        "stored_path": str(stored_path),
        "applied_job_ids": applied_job_ids,
        "results": {},   # job_id -> résultat d'évaluation sérialisé
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }
    candidates = load_candidates()
    candidates.append(candidate)
    save_candidates(candidates)
    return candidate


def candidates_for_job(job_id: str) -> list[dict]:
    return [c for c in load_candidates() if job_id in c.get("applied_job_ids", [])]


def save_result_for_candidate(candidate_id: str, job_id: str, result: dict) -> None:
    candidates = load_candidates()
    for c in candidates:
        if c["id"] == candidate_id:
            c.setdefault("results", {})[job_id] = result
            break
    save_candidates(candidates)


def delete_all_data() -> None:
    """Repart de zéro (utile en dev / démo)."""
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    _ensure_dirs()
