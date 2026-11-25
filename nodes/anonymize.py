"""Masque les données personnelles avant tout appel LLM / scoring."""
from __future__ import annotations
import re

_nlp = None

_NOT_A_NAME = re.compile(
    r"(?i)\b(curriculum|vitae|cv|resume|résumé|profil|profile|contact|"
    r"étudiant|étudiante|ingénieur|développeur|stagiaire)\b"
)
_NAME_LINE = re.compile(r"[A-ZÀ-Ý][\w'’-]+(?: [A-ZÀ-Ý][\w'’-]+){1,3}")


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("fr_core_news_md", disable=["parser", "lemmatizer"])
    return _nlp


EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Téléphone : on exclut explicitement les périodes (2022-2024, 2022 – 2024).
PHONE = re.compile(
    r"(?<![\w/])(?!(?:19|20)\d{2}\s?[-–]\s?(?:19|20)\d{2})"
    r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){8,11}(?!\d)"
)
PATTERNS = [
    ("EMAIL", EMAIL),
    ("URL", re.compile(r"(?i)(?:https?://|www\.|linkedin\.com/|github\.com/)\S+")),
    ("PHONE", PHONE),
    ("NAISSANCE", re.compile(r"(?i)\b(?:né(?:e)?\s+le|date\s+de\s+naissance|born)\s*:?\s*[^\n]{0,30}")),
    ("AGE", re.compile(r"(?i)\b(?:âge|age)\s*:?\s*\d{2}\s*ans|\b\d{2}\s*ans\b(?!\s*(?:d['’]|de\s))")),
    ("GENRE", re.compile(r"(?i)\b(?:sexe|genre)\s*:?\s*(?:homme|femme|masculin|féminin|[MF])\b")),
    ("CIVILITE", re.compile(r"\b(?:Mme|Mlle|Madame|Monsieur)\b|\bM\.(?=\s)")),
    ("SITUATION", re.compile(r"(?i)\b(?:situation familiale|nationalité|marié\(?e?\)?|célibataire)\s*:?[^\n]{0,25}")),
    ("ADRESSE", re.compile(r"(?i)\b\d{1,3}[, ]+(?:rue|avenue|av\.|boulevard|bd|impasse|allée|route|cité)\s+[^\n]{3,50}")),
]


def _detect_name(text: str) -> str | None:
    """1) première ligne qui ressemble à « Prénom Nom », 2) sinon NER sur l'en-tête."""
    for line in text.splitlines()[:6]:
        line = line.strip()
        if _NAME_LINE.fullmatch(line) and not _NOT_A_NAME.search(line):
            return line
    try:
        doc = _get_nlp()(text[:400])
    except Exception:
        return None
    for ent in doc.ents:
        name = ent.text.strip()
        if ent.label_ == "PER" and 2 <= len(name.split()) <= 4 and not _NOT_A_NAME.search(name):
            return name
    return None


def anonymize(text: str) -> tuple[str, dict]:
    email_match = EMAIL.search(text)
    contact = {
        "email": email_match.group(0).rstrip(".") if email_match else None,
        "full_name": _detect_name(text),
    }

    masked = text
    for label, pattern in PATTERNS:
        masked = pattern.sub(f"[{label}]", masked)

    # Masque uniquement le nom détecté (et ses parties), pas toutes les entités PER :
    # sinon des noms d'outils ou d'entreprises seraient masqués par erreur.
    name = contact["full_name"]
    if name:
        parts = [name] + [p for p in re.split(r"[\s-]+", name) if len(p) >= 3]
        for part in sorted(set(parts), key=len, reverse=True):
            masked = re.sub(rf"(?i)\b{re.escape(part)}\b", "[NOM]", masked)

    return masked, contact


def anonymize_node(state: dict) -> dict:
    masked, contact = anonymize(state["extracted_text"])
    return {"anonymized_text": masked, "contact_info": contact}