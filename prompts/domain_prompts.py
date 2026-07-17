"""
Premier jet de contexte métier par domaine — à ajuster selon vos besoins réels.

Objectif : sans ce contexte, le nœud skill_relevance (LLM) doit deviner tout seul
qu'"analyse de données" veut dire "Power BI/Excel" en finance mais "Python/SQL/Spark"
en tech. Ce contexte lui donne un vocabulaire métier de départ pour chaque domaine, ce
qui réduit les faux "0% de compétences en commun" quand une offre est formulée en
termes généraux.

Ce n'est PAS une checklist figée : le LLM garde la liberté de juger au-delà de cette
liste si le CV/l'offre contiennent des éléments non couverts ici. C'est un point de
départ, pas une contrainte stricte.
"""
from __future__ import annotations

DOMAIN_SKILL_CONTEXT: dict[str, str] = {
    "commerce": (
        "Domaine Commerce/Vente. Repères de vocabulaire : \"analyse de données\" désigne ici "
        "le plus souvent le suivi de KPIs commerciaux (taux de conversion, panier moyen, pipe "
        "de vente) via un CRM (Salesforce, HubSpot, Pipedrive) ou Excel, pas de la data "
        "science. \"Développement commercial\" couvre prospection, négociation, closing, "
        "gestion de compte, cycle de vente B2B/B2C. \"Outils de vente\" = CRM, LinkedIn Sales "
        "Navigator, outils d'emailing (Lemlist, Salesloft). \"Marketing\" adjacent peut inclure "
        "growth, SEO/SEA, réseaux sociaux — à ne pas confondre avec un poste purement commercial "
        "sauf si l'offre le précise."
    ),
    "tech": (
        "Domaine Tech/Ingénierie. Repères de vocabulaire : \"analyse de données\" désigne "
        "typiquement Python, SQL, Spark, pandas, ou des plateformes data (Airflow, dbt, "
        "Snowflake, BigQuery). \"Développement\" couvre les langages (Python, Java, "
        "JavaScript/TypeScript, Go, C++) et frameworks associés (React, Django, Spring). "
        "\"Cloud\" = AWS/Azure/GCP et leurs services managés. \"DevOps\" = CI/CD, Docker, "
        "Kubernetes, Terraform. \"Connaissances de base en programmation\" est généralement "
        "satisfait par la maîtrise d'au moins un langage courant, même sans intitulé de poste "
        "dev pur."
    ),
    "finance": (
        "Domaine Finance/Comptabilité. Repères de vocabulaire : \"analyse de données\" "
        "désigne ici le plus souvent Excel avancé (TCD, modélisation financière, VBA), "
        "Power BI, ou du reporting sur un ERP (SAP, Oracle) — rarement Python/SQL sauf mention "
        "explicite (poste \"data\" en finance quantitative). \"Modélisation financière\" = "
        "construction de business plans, prévisionnels, valorisation. \"Comptabilité\" = "
        "normes (IFRS, PCG), clôtures, consolidation. \"Réglementaire\" = conformité, Bâle "
        "III, contrôle de gestion."
    ),
    "operations": (
        "Domaine Opérations/Supply Chain. Repères de vocabulaire : \"analyse de données\" "
        "désigne le suivi d'indicateurs logistiques (taux de service, rotation de stock, "
        "délais) via Excel ou un ERP/WMS (SAP, Oracle SCM), rarement du code. "
        "\"Amélioration continue\" = Lean, Six Sigma, Kaizen. \"Planification\" = gestion de "
        "stocks, prévision de la demande, S&OP. \"Qualité\" = normes ISO, audits, gestion des "
        "non-conformités."
    ),
}

DOMAIN_LABELS = {
    "commerce": "Commerce",
    "tech": "Tech",
    "finance": "Finance",
    "operations": "Opérations",
}


def get_domain_context(domain: str | None) -> str:
    """Renvoie le contexte métier pour un domaine donné, ou une chaîne vide si inconnu."""
    if not domain:
        return ""
    return DOMAIN_SKILL_CONTEXT.get(domain, "")
