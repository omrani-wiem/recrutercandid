"""
Authentification simple par comptes stockés localement (data/users.json).
Mots de passe hashés (PBKDF2 + sel), jamais stockés en clair.

Comptes de démo créés au premier lancement (rôles "recruteur" et "etudiant").
Les mots de passe viennent de DEMO_RECRUITER_PASSWORD et DEMO_STUDENT_PASSWORD
(voir env.example). S'ils ne sont pas définis, un mot de passe aléatoire est généré
et affiché une seule fois dans la console : aucun mot de passe n'est écrit en dur.

Extensible : add_user() permet d'ajouter d'autres comptes plus tard (ex: page d'admin,
ou inscription libre côté étudiant).
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
USERS_PATH = DATA_DIR / "users.json"

ROLE_RECRUTEUR = "recruteur"
ROLE_ETUDIANT = "etudiant"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def _load_users() -> dict:
    _ensure_dirs()
    if not USERS_PATH.exists():
        users = _seed_default_users()
        _save_users(users)
        return users
    with USERS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_users(users: dict) -> None:
    _ensure_dirs()
    with USERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _demo_password(env_var: str, username: str) -> str:
    """Mot de passe d'un compte de démo : variable d'environnement, sinon aléatoire."""
    password = os.getenv(env_var, "").strip()
    if password:
        return password
    password = secrets.token_urlsafe(12)
    print(f"[auth] Compte de démo '{username}' créé. Mot de passe généré (affiché une seule fois) : {password}")
    return password


def _seed_default_users() -> dict:
    return {
        "recruteur": _make_user_entry(_demo_password("DEMO_RECRUITER_PASSWORD", "recruteur"), ROLE_RECRUTEUR),
        "etudiant": _make_user_entry(_demo_password("DEMO_STUDENT_PASSWORD", "etudiant"), ROLE_ETUDIANT),
    }


def _make_user_entry(password: str, role: str) -> dict:
    salt = os.urandom(16)
    return {
        "salt": salt.hex(),
        "password_hash": _hash_password(password, salt),
        "role": role,
    }


def add_user(username: str, password: str, role: str) -> bool:
    """Crée un nouveau compte. Renvoie False si le nom d'utilisateur existe déjà."""
    users = _load_users()
    if username in users:
        return False
    users[username] = _make_user_entry(password, role)
    _save_users(users)
    return True


def verify_login(username: str, password: str) -> str | None:
    """Vérifie les identifiants. Renvoie le rôle si valide, sinon None."""
    users = _load_users()
    entry = users.get(username)
    if not entry:
        return None
    salt = bytes.fromhex(entry["salt"])
    if hmac.compare_digest(_hash_password(password, salt), entry["password_hash"]):
        return entry["role"]
    return None