"""
Schéma d'état du pipeline. C'est la seule structure qui transite entre les nœuds :
- les nœuds fonction la lisent/mutent directement
- les nœuds LLM ne reçoivent JAMAIS le texte brut du CV au-delà de leur propre étape :
  une fois extrait en JSON structuré, le texte brut n'est plus repropagé (surface d'attaque
  pour le prompt injection réduite au strict minimum : un seul appel LLM voit le texte brut).
"""
from __future__ import annotations
from typing import TypedDict, Optional, Literal
from pydantic import BaseModel, Field


class ResumeProfile(BaseModel):
    full_name: str
    email: Optional[str] = None
    years_experience: float = 0
    skills: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    raw_sections: dict[str, str] = Field(default_factory=dict)


class JobRequirements(BaseModel):
    title: str
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0
    required_certifications: list[str] = Field(default_factory=list)
    location: Optional[str] = None


class MatchScore(BaseModel):
    # métriques principales
    semantic_similarity: float
    skill_overlap_ratio: float
    experience_match: float
    hard_filters_passed: bool
    final_score: float  # score sur 20

    # détails d'explication
    candidate_years_experience: float = 0.0
    required_years_experience: float = 0.0

    # diagnostic
    failed_filters: list[str] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)

    # transparence : comment le recoupement de compétences a été établi.
    # "keyword"      -> correspondance mot-à-mot / synonymes (rapide, déterministe, limité)
    # "llm_assisted" -> un LLM a jugé qu'une compétence concrète du candidat satisfait une
    #                   exigence formulée de façon générale (voir nodes/skill_relevance.py)
    skill_matching_method: Literal["keyword", "llm_assisted"] = "keyword"

    # breakdown exploitable par l'UI
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class SecurityFlag(BaseModel):
    node: str
    severity: Literal["info", "warning", "blocked"]
    reason: str


class PipelineState(TypedDict, total=False):
    # entrées brutes
    cv_path: str
    jd_text: str
    domain: Optional[str]  # "commerce" | "tech" | "finance" | "operations" (voir prompts/domain_prompts.py)

    # sorties OCR
    extracted_text: str
    ocr_confidence: float
    ocr_method: Literal["native", "qwen_vlm", "qwen_vlm_failed"]
    ocr_error: Optional[str]

    # sorties structurées
    resume: ResumeProfile
    job: JobRequirements

    # jugement de correspondance sémantique des compétences (assisté par LLM, optionnel).
    # Voir nodes/skill_relevance.py : ne décide jamais du score final, ne fait que juger
    # si une compétence concrète du candidat satisfait une exigence formulée en termes
    # généraux (ex: "Python" satisfait "compétences en analyse de données").
    skill_relevance_judgment: Optional[dict]

    # matching
    score: MatchScore
    decision: Literal["shortlist", "reject", "needs_human_review"]

    # explication / communication
    explanation: str
    rejection_email_draft: dict
    rejection_email_approved: bool
    rejection_email_sent: bool
    email_send_error: Optional[str]

    # sécurité / observabilité
    security_flags: list[SecurityFlag]
    trace_id: str
