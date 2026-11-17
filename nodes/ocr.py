"""
Nœud FONCTION (pas de LLM ici) : extraction de texte natif depuis le PDF, puis évaluation
heuristique de la qualité pour décider si on doit basculer sur Qwen VLM (nœud séparé, appelé
via le serveur MCP mcp_servers/ocr_mcp_server.py).
"""
from __future__ import annotations
import re


def extract_native_text(cv_path: str) -> str:
    """Extraction texte natif. pdfplumber gère la majorité des CV générés numériquement."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(cv_path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def compute_confidence(text: str) -> float:
    """
    Score heuristique simple, pas besoin de LLM :
    - longueur du texte extrait
    - ratio de caractères imprimables / total
    - présence de sections attendues d'un CV
    """
    if not text or len(text.strip()) < 50:
        return 0.0

    printable_ratio = sum(c.isprintable() for c in text) / max(len(text), 1)

    expected_sections = ["expérience", "experience", "compétence", "skill",
                          "formation", "education", "diplôme"]
    section_hits = sum(1 for s in expected_sections if s.lower() in text.lower())
    section_score = min(section_hits / 2, 1.0)

    garbled_ratio = len(re.findall(r"[^\x00-\x7F\u00C0-\u017F]{3,}", text)) / max(
        len(text) / 200, 1
    )
    garbled_penalty = max(0, 1 - garbled_ratio * 0.3)

    confidence = 0.4 * printable_ratio + 0.4 * section_score + 0.2 * garbled_penalty
    return round(min(confidence, 1.0), 2)


def ocr_node(state: dict) -> dict:
    """Nœud LangGraph : tente l'extraction native, calcule la confiance."""
    text = extract_native_text(state["cv_path"])
    confidence = compute_confidence(text)
    return {
        "extracted_text": text,
        "ocr_confidence": confidence,
        "ocr_method": "native",
    }


def needs_vlm_fallback(state: dict) -> str:
    """Fonction de branchement conditionnel LangGraph (add_conditional_edges)."""
    threshold = 0.55
    if state.get("ocr_confidence", 0) < threshold:
        return "qwen_vlm_ocr"
    return "extraction"
