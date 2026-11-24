from io import BytesIO
from datetime import datetime
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_summary_pdf(name: str, job: dict, result: dict, anonymous: bool = False) -> bytes:
    styles = getSampleStyleSheet()
    score, resume = result.get("score") or {}, result.get("resume") or {}
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Dossier de synthèse")
    P = lambda t, s="BodyText": Paragraph(escape(str(t)), styles[s])

    story = [
        P("Dossier de synthèse candidat", "Title"),
        P(f"Poste : {job['title']}"),
        P(f"Candidat : {'Anonyme' if anonymous else name}"),
        P(f"Généré le {datetime.now():%d/%m/%Y %H:%M}"),
        Spacer(1, 12),
    ]
    table = Table([
        ["Score final", f"{score.get('final_score', '-')} / 20"],
        ["Compétences en commun", f"{round(score.get('skill_overlap_ratio', 0) * 100, 1)} %"],
        ["Expérience", f"{resume.get('years_experience', 0):.0f} an(s)"],
        ["Conditions obligatoires", "Respectées" if score.get("hard_filters_passed") else "Non respectées"],
    ], colWidths=[180, 250])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
    ]))
    story += [table, Spacer(1, 12)]
    story += [P("Compétences en commun", "Heading2"), P(", ".join(score.get("matched_skills", [])) or "aucune")]
    story += [P("Compétences manquantes", "Heading2"), P(", ".join(score.get("missing_skills", [])) or "aucune")]
    story += [P("Analyse", "Heading2"), P(result.get("explanation") or "Aucune explication.")]
    story += [Spacer(1, 12), P("Document d'aide à la décision : la décision finale revient au recruteur.", "Italic")]
    doc.build(story)
    return buf.getvalue()