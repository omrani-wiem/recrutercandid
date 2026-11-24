"""Masque les données personnelles avant tout appel LLM / scoring."""
from __future__ import annotations
import re

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("fr_core_news_md")
    return _nlp

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\d[\s.-]?){8,11}(?!\d)")
PATTERNS = [
    ("EMAIL", EMAIL),
    ("URL", re.compile(r"(?i)(?:https?://|www\.|linkedin\.com/|github\.com/)\S+")),
    ("PHONE", PHONE),
    ("NAISSANCE", re.compile(r"(?i)\b(?:né(?:e)?\s+le|date\s+de\s+naissance|born)\s*:?\s*[^\n]{0,30}")),
    ("AGE", re.compile(r"(?i)\b(?:âge|age)\s*:?\s*\d{2}\s*ans|\b\d{2}\s*ans\b(?!\s*(?:d['’]|de\s))")),
    ("GENRE", re.compile(r"(?i)\b(?:sexe|genre)\s*:?\s*(?:homme|femme|masculin|féminin|[MF])\b")),
    ("CIVILITE", re.compile(r"\b(?:Mme|Mlle|Madame|Monsieur|M\.)\s")),
    ("SITUATION", re.compile(r"(?i)\b(?:situation familiale|nationalité|marié\(?e?\)?|célibataire)\s*:?[^\n]{0,25}")),
    ("ADRESSE", re.compile(r"(?i)\b\d{1,3}[, ]+(?:rue|avenue|av\.|boulevard|bd|impasse|allée|route|cité)\s+[^\n]{3,50}")),
]

def anonymize(text: str) -> tuple[str, dict]:
    contact = {
        "email": (EMAIL.search(text) or [None])[0] if EMAIL.search(text) else None,
        "full_name": None,
    }
    nlp = _get_nlp()
    head = nlp(text[:400])
    for ent in head.ents:
        if ent.label_ == "PER":
            contact["full_name"] = ent.text.strip()
            break

    masked = text
    for label, pattern in PATTERNS:
        masked = pattern.sub(f"[{label}]", masked)

    doc = nlp(masked)
    for ent in reversed(doc.ents):
        if ent.label_ == "PER":
            masked = masked[:ent.start_char] + "[NOM]" + masked[ent.end_char:]
    return masked, contact


def anonymize_node(state: dict) -> dict:
    masked, contact = anonymize(state["extracted_text"])
    return {"anonymized_text": masked, "contact_info": contact}

