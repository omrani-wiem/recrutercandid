from __future__ import annotations

import os
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

import data_store
from graph import build_graph
from ui_helpers import candidates_to_analyze
from pipeline_runner import (
    run_analysis, resume_decision, resume_email_review,
    STATUS_SHORTLISTED, STATUS_PENDING_DECISION, STATUS_PENDING_EMAIL_REVIEW,
    STATUS_REJECTED_EMAIL_SENT, STATUS_REJECTED_EMAIL_CANCELLED,
    STATUS_REJECTED_EMAIL_FAILED, STATUS_ERROR,
)
from rl.feedback_store import append_feedback, load_feedback_log, clear_feedback_log
from rl.scoring_model import offline_update

import auth
from job_search import search_jobs_abroad, ADZUNA_COUNTRIES

from cv_advisor import generate_cv_improvement_report
from nodes.ocr import extract_native_text


load_dotenv()
st.set_page_config(page_title="Tri de CV", page_icon="📄", layout="wide")

STATUS_LABELS = {
    STATUS_SHORTLISTED: (" Retenu", "success"),
    STATUS_PENDING_DECISION: (" Décision à prendre", "warning"),
    STATUS_PENDING_EMAIL_REVIEW: (" Email à valider", "warning"),
    STATUS_REJECTED_EMAIL_SENT: (" Refusé, email envoyé", "info"),
    STATUS_REJECTED_EMAIL_CANCELLED: ("Refusé, email annulé", "info"),
    STATUS_REJECTED_EMAIL_FAILED: (" Refusé, email non envoyé", "warning"),
    STATUS_ERROR: (" Erreur", "error"),
    None: ("Non évalué", "off"),
}

# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------
if "graph_app" not in st.session_state:
    st.session_state.graph_app = build_graph()
if "view" not in st.session_state:
    st.session_state.view = "domains"
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = None
if "selected_job_id" not in st.session_state:
    st.session_state.selected_job_id = None
if "selected_candidate_id" not in st.session_state:
    st.session_state.selected_candidate_id = None
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None


def go(view: str, **kwargs) -> None:
    st.session_state.view = view
    for k, v in kwargs.items():
        st.session_state[k] = v
    st.rerun()


def candidate_display_name(candidate: dict, job_id: str | None = None) -> str:
    if job_id:
        res = candidate.get("results", {}).get(job_id)
        if res and res.get("resume") and res["resume"].get("full_name"):
            name = res["resume"]["full_name"]
            if "INCONNU" not in name:
                return name
    return candidate["original_filename"]

def delete_analysis(candidate_id: str, job_id: str) -> None:
    thread_id = f"{candidate_id}::{job_id}"
    try:
        st.session_state.graph_app.checkpointer.delete_thread(thread_id)
    except Exception:
        pass  # pas grave si rien à supprimer côté checkpointer
    data_store.delete_result_for_candidate(candidate_id, job_id)


def delete_candidate_everywhere(candidate: dict) -> None:
    for job_id in candidate.get("applied_job_ids", []):
        try:
            st.session_state.graph_app.checkpointer.delete_thread(f"{candidate['id']}::{job_id}")
        except Exception:
            pass
    data_store.delete_candidate(candidate["id"])


def status_badge(status: str | None):
    label, kind = STATUS_LABELS.get(status, (status or "Non évalué", "off"))
    if kind == "success":
        st.success(label)
    elif kind == "warning":
        st.warning(label)
    elif kind == "error":
        st.error(label)
    elif kind == "info":
        st.info(label)
    else:
        st.caption(label)


# ---------------------------------------------------------------------------
# Écran de connexion
# ---------------------------------------------------------------------------
def render_login():
    st.title("📄 Tri de CV — Connexion")
    with st.form("login_form"):
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter", type="primary")
        if submitted:
            role = auth.verify_login(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.session_state.view = "domains" if role == auth.ROLE_RECRUTEUR else "job_search"
                st.rerun()
            else:
                st.error("Identifiants incorrects.")
    st.caption("Comptes de démo : « recruteur » et « etudiant » (mots de passe définis dans .env)")


if not st.session_state.authenticated:
    render_login()
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar : vivier de CV + amélioration du score
# ---------------------------------------------------------------------------
with st.sidebar:
    st.caption(f"Connecté en tant que **{st.session_state.username}** ({st.session_state.role})")
    if st.button("🚪 Déconnexion"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.role = None
        st.session_state.view = "domains"
        st.rerun()
    st.divider()
    if st.session_state.role == auth.ROLE_RECRUTEUR:
     st.header("Vivier de CV")
     st.caption("Un CV ajouté ici est automatiquement candidat à toutes les offres existantes.")
     cv_files = st.file_uploader("Ajouter des CV (PDF)", type=["pdf"], accept_multiple_files=True, key="cv_uploader")
     if cv_files and st.button("Ajouter au vivier"):
         for f in cv_files:
             data_store.add_candidate_cv(f.getvalue(), f.name, apply_to_all_jobs=True)
         st.success(f"{len(cv_files)} CV ajouté(s) au vivier.")
         st.rerun()

     st.divider()
     st.header("Amélioration du score")
     st.caption(
         "Chaque avis 👍/👎 donné sur un candidat est mis de côté ici. Cliquez pour que "
         "l'outil en tienne compte pour les prochaines évaluations."
     )
     pending_feedback = load_feedback_log()
     st.caption(f"{len(pending_feedback)} avis en attente de prise en compte.")
     if st.button("Tenir compte de mes avis"):
         if not pending_feedback:
             st.warning("Aucun avis pour l'instant.")
         else:
             offline_update(pending_feedback)
             clear_feedback_log()
             st.success(f"{len(pending_feedback)} avis pris en compte.")

     if st.session_state.view != "domains":
         st.divider()
         if st.button("🏠 Retour à l'accueil"):
             go("domains")


# ---------------------------------------------------------------------------
# Vue 1 : choix du domaine
# ---------------------------------------------------------------------------
def render_domains():
    st.title("Tri de CV")
    st.caption("Choisissez un domaine pour voir ses offres et les candidats associés.")

    cols = st.columns(len(data_store.DOMAINS))
    for col, domain in zip(cols, data_store.DOMAINS):
        with col:
            with st.container(border=True):
                st.subheader(data_store.DOMAIN_LABELS[domain])
                n_jobs = len(data_store.jobs_for_domain(domain))
                st.caption(f"{n_jobs} offre(s)")
                if st.button("Voir les offres", key=f"domain_{domain}", use_container_width=True):
                    go("jobs", selected_domain=domain)


# ---------------------------------------------------------------------------
# Vue 2 : liste des offres d'un domaine
# ---------------------------------------------------------------------------
def render_jobs():
    domain = st.session_state.selected_domain
    st.title(f"Offres — {data_store.DOMAIN_LABELS[domain]}")

    jobs = data_store.jobs_for_domain(domain)
    if not jobs:
        st.info("Aucune offre dans ce domaine pour l'instant.")
    for job in jobs:
        with st.container(border=True):
            st.subheader(job["title"])
            st.caption(job.get("location", "") or "Lieu non renseigné")
            st.write(job["description"][:220] + ("…" if len(job["description"]) > 220 else ""))
            n_candidates = len(data_store.candidates_for_job(job["id"]))
            col1, col2 = st.columns([3, 1])
            col1.caption(f"{n_candidates} candidat(s) ont postulé")
            if col2.button("Voir les candidats →", key=f"job_{job['id']}"):
                go("job_detail", selected_job_id=job["id"])

    with st.expander("➕ Ajouter une offre"):
        title = st.text_input("Intitulé du poste", key="new_job_title")
        location = st.text_input("Lieu", key="new_job_location")
        description = st.text_area(
            "Description complète de l'offre (compétences requises, expérience minimale, etc.)",
            height=180, key="new_job_description",
        )
        if st.button("Créer l'offre"):
            if not title or not description:
                st.error("Le titre et la description sont obligatoires.")
            else:
                data_store.add_job(domain, title, description, location)
                st.success("Offre créée.")
                st.rerun()


# ---------------------------------------------------------------------------
# Vue 3 : détail d'une offre + liste des candidats + lancement de l'analyse
# ---------------------------------------------------------------------------
def render_job_detail():
    job = data_store.get_job(st.session_state.selected_job_id)
    if not job:
        st.error("Offre introuvable.")
        return
    st.title(job["title"])
    st.caption(f"{data_store.DOMAIN_LABELS[job['domain']]} · {job.get('location', 'Lieu non renseigné')}")
    with st.expander("Description de l'offre", expanded=False):
        st.write(job["description"])

    candidates = data_store.candidates_for_job(job["id"])
    st.subheader(f"{len(candidates)} candidat(s)")

    unanalyzed = candidates_to_analyze(candidates, job["id"])

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            f"🔎 Analyser / relancer les candidats à traiter ({len(unanalyzed)})",
            type="primary",
            disabled=not unanalyzed,
        ):
            graph_app = st.session_state.graph_app
            progress = st.progress(0.0, text="Analyse en cours...")
            for i, cand in enumerate(unanalyzed):
                result = run_analysis(graph_app, cand, job)
                data_store.save_result_for_candidate(cand["id"], job["id"], result)
                progress.progress((i + 1) / len(unanalyzed), text=f"Analyse en cours... ({i + 1}/{len(unanalyzed)})")
            progress.empty()
            st.rerun()
    with col2:
        if st.button("📊 Voir le classement"):
            go("ranking", selected_job_id=job["id"])

    st.divider()
    for cand in candidates:
        res = cand.get("results", {}).get(job["id"])
        status = res["status"] if res else None
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"**{candidate_display_name(cand, job['id'])}**")
            with c2:
                status_badge(status)
            if res and res.get("score"):
                c1.caption(f"Score : {res['score']['final_score']}/20")
            if c3.button("Détail", key=f"detail_{cand['id']}"):
                go("candidate_detail", selected_candidate_id=cand["id"])
            if res and c3.button("🔄 Réinitialiser", key=f"delete_{cand['id']}"):
                delete_analysis(cand["id"], job["id"])
                st.rerun()
            if c3.button("🗑️ Retirer le candidat", key=f"remove_{cand['id']}"):
                delete_candidate_everywhere(cand)
                st.rerun()
            if res and res.get("status") == "error":
                st.caption(f"Erreur d'analyse : {res.get('error', 'cause inconnue')}")
# ---------------------------------------------------------------------------
# Vue 4 : classement
# ---------------------------------------------------------------------------
def render_ranking():
    job = data_store.get_job(st.session_state.selected_job_id)
    if not job:
        st.error("Offre introuvable.")
        return
    st.title(f"Classement — {job['title']}")
    if st.button("← Retour à l'offre"):
        go("job_detail", selected_job_id=job["id"])

    candidates = data_store.candidates_for_job(job["id"])
    scored = []
    unscored_count = 0
    for cand in candidates:
        res = cand.get("results", {}).get(job["id"])
        if res and res.get("score"):
            scored.append((cand, res))
        else:
            unscored_count += 1
    scored.sort(key=lambda pair: pair[1]["score"]["final_score"], reverse=True)

    if unscored_count:
        st.caption(f"{unscored_count} candidat(s) pas encore analysé(s) — non affichés ici.")

    for rank, (cand, res) in enumerate(scored, start=1):
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([0.5, 3, 1.5, 1])
            c1.write(f"**#{rank}**")
            c2.write(f"**{candidate_display_name(cand, job['id'])}**")
            c3.metric("Score", f"{res['score']['final_score']}/20", label_visibility="collapsed")
            if c4.button("Détail", key=f"rank_detail_{cand['id']}"):
                go("candidate_detail", selected_candidate_id=cand["id"])


# ---------------------------------------------------------------------------
# Vue 5 : détail candidat (score, décision, email, feedback)
# ---------------------------------------------------------------------------
def render_candidate_detail():
    candidate_id = st.session_state.selected_candidate_id
    candidates = data_store.load_candidates()
    candidate = next((c for c in candidates if c["id"] == candidate_id), None)
    if not candidate:
        st.error("Candidat introuvable.")
        return

    job = data_store.get_job(st.session_state.selected_job_id)
    if not job:
        st.error("Offre introuvable.")
        return

    if st.button("← Retour"):
        go("job_detail", selected_job_id=job["id"])
    
   
    result = candidate.get("results", {}).get(job["id"])
    if not result:
        st.info("Ce candidat n'a pas encore été analysé pour cette offre.")
        if st.button("Analyser maintenant"):
            result = run_analysis(st.session_state.graph_app, candidate, job)
            data_store.save_result_for_candidate(candidate["id"], job["id"], result)
            st.rerun()
        return
    
    if st.button("🗑️ Supprimer cette analyse"):
        delete_analysis(candidate["id"], job["id"])
        go("job_detail", selected_job_id=job["id"])

    resume = result.get("resume")
    score = result.get("score")
    status = result.get("status")

    st.title(candidate_display_name(candidate, job["id"]))
    status_badge(status)

    if status == STATUS_ERROR:
        st.error(f"Erreur d'analyse : {result.get('error', 'cause inconnue')}")
        return

    if result.get("ocr_error"):
        st.warning(
            "Ce CV n'a pas pu être lu correctement par l'outil (mise en page complexe ou "
            "fichier scanné). Vérifiez son contenu manuellement avant de vous fier au score."
        )

    if score:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score de compatibilité", f"{score['final_score']} / 20")
        col2.metric(
            "Compétences en commun",
            f"{round(score['skill_overlap_ratio'] * 100, 1)} %",
            help=(
                "Évaluée par l'IA (comprend les formulations générales de l'offre)."
                if score.get("skill_matching_method") == "llm_assisted"
                else "Basée sur une correspondance de mots-clés exacts."
            ),
        )
        col3.metric(
            "Expérience",
            "Suffisante" if score["candidate_years_experience"] >= score["required_years_experience"] else "Insuffisante",
        )
        col4.metric("Conditions obligatoires", "Respectées" if score["hard_filters_passed"] else "Non respectées")

        if (
            score.get("skill_matching_method") == "keyword"
            and score["skill_overlap_ratio"] == 0
            and (score["candidate_years_experience"] > 0 or score["hard_filters_passed"])
        ):
            st.info(
                "ℹ️ « Compétences en commun : 0 % » ne veut pas forcément dire que le candidat "
                "n'est pas qualifié — l'offre ne cite peut-être pas de technologies précises. "
                "Fiez-vous aussi à la liste de compétences ci-dessous."
            )

        st.subheader("Pourquoi ce score ?")
        st.write(result.get("explanation") or "Aucune explication disponible.")

        with st.expander("Voir le détail des compétences"):
            st.write(f"**En commun :** {', '.join(score['matched_skills']) or 'aucune'}")
            st.write(f"**Manquantes :** {', '.join(score['missing_skills']) or 'aucune'}")
            if score.get("failed_filters"):
                st.write(f"**Conditions non respectées :** {', '.join(score['failed_filters'])}")

    if resume:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Profil du candidat")
            st.write(f"**Nom :** {resume['full_name']}")
            st.write(f"**Expérience :** {resume['years_experience']:.0f} an(s)")
            st.write(f"**Compétences :** {', '.join(resume['skills']) or 'Aucune détectée'}")
        with col_b:
            st.subheader("Offre")
            st.write(f"**Poste :** {job['title']}")
            st.write(f"**Lieu :** {job.get('location') or 'Non renseigné'}")

    st.divider()

    # --- Décision en attente (zone grise) ---
    if status == STATUS_PENDING_DECISION:
        st.subheader("Décision")
        st.caption("Ce score est en zone grise : à vous de trancher.")
        c1, c2 = st.columns(2)
        if c1.button("✅ Shortlister", type="primary"):
            result = resume_decision(st.session_state.graph_app, candidate, job, "shortlist")
            data_store.save_result_for_candidate(candidate["id"], job["id"], result)
            st.rerun()
        if c2.button("❌ Rejeter"):
            result = resume_decision(st.session_state.graph_app, candidate, job, "reject")
            data_store.save_result_for_candidate(candidate["id"], job["id"], result)
            st.rerun()

    # --- Email de refus en attente d'approbation ---
    if status == STATUS_PENDING_EMAIL_REVIEW and result.get("rejection_email_draft"):
        st.subheader("Email de refus")
        draft = result["rejection_email_draft"]
        subject = st.text_input("Objet", value=draft.get("subject", ""), key=f"subj_{candidate_id}")
        body = st.text_area("Message", value=draft.get("body", ""), height=260, key=f"body_{candidate_id}")

        c1, c2, c3 = st.columns(3)
        if c1.button("Envoyer", type="primary"):
            edited = (subject != draft.get("subject", "") or body != draft.get("body", ""))
            result = resume_email_review(
                st.session_state.graph_app, candidate, job,
                action="edit" if edited else "approve",
                edited_subject=subject, edited_body=body,
            )
            data_store.save_result_for_candidate(candidate["id"], job["id"], result)
            st.rerun()
        if c2.button("Ne pas envoyer"):
            result = resume_email_review(st.session_state.graph_app, candidate, job, action="cancel")
            data_store.save_result_for_candidate(candidate["id"], job["id"], result)
            st.rerun()

    if status == STATUS_REJECTED_EMAIL_FAILED:
        st.error(f"L'envoi a échoué : {result.get('email_send_error')}")

    # --- Avis recruteur (feedback pour recalibrage du score) ---
    if score:
        st.divider()
        st.subheader("Votre avis sur ce candidat")
        feedback_key = f"{candidate_id}::{job['id']}"
        feedback_state = st.session_state.feedback_given.get(feedback_key)
        c1, c2 = st.columns(2)
        mark_good = c1.button("👍 Bon profil pour ce poste", key=f"fb_good_{feedback_key}")
        mark_bad = c2.button("👎 Ne correspond pas au poste", key=f"fb_bad_{feedback_key}")
        if mark_good or mark_bad:
            append_feedback(
                features=score["score_breakdown"],
                reward=1.0 if mark_good else 0.0,
                candidate_name=candidate_display_name(candidate, job["id"]),
                job_title=job["title"],
            )
            st.session_state.feedback_given[feedback_key] = "positif" if mark_good else "négatif"
            feedback_state = st.session_state.feedback_given[feedback_key]
        if feedback_state:
            st.info(f"Avis enregistré ({feedback_state}). Merci !")


# ---------------------------------------------------------------------------
# Vue étudiant : recherche d'offres à l'étranger
# ---------------------------------------------------------------------------
def render_job_search():
    st.title("🔍 Recherche d'offres à l'étranger")
    st.caption("Recherche en temps réel via Adzuna et Arbeitnow — uniquement des offres actives.")

    with st.expander("📄 Mon CV (pour obtenir un rapport d'amélioration par offre)", expanded=False):
        cv_file = st.file_uploader("Téléverser mon CV (PDF)", type=["pdf"], key="student_cv")
        if cv_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(cv_file.getvalue())
                tmp_path = tmp.name
            st.session_state["student_cv_text"] = extract_native_text(tmp_path)
            st.success("CV chargé. Tu peux maintenant demander un rapport sur chaque offre ci-dessous.")

    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Poste recherché", placeholder="ex: data analyst, développeur web...")
    with col2:
        country_label = st.selectbox(
            "Pays (Adzuna)",
            options=["Remote / International uniquement"] + list(ADZUNA_COUNTRIES.values()),
        )

    if st.button("Rechercher", type="primary", disabled=not query):
        country_code = None
        if country_label != "Remote / International uniquement":
            country_code = next(k for k, v in ADZUNA_COUNTRIES.items() if v == country_label)
        with st.spinner("Recherche en cours..."):
            results = search_jobs_abroad(query, adzuna_country=country_code)
        st.session_state["job_search_results"] = results

    results = st.session_state.get("job_search_results", [])
    if results:
        st.subheader(f"{len(results)} offre(s) trouvée(s)")
        for i, job in enumerate(results):
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{job['title']}** — {job['company']}")
                c1.caption(f"📍 {job['location'] or job['country']} · Source : {job['source']}")
                if job.get("description"):
                    with c1.expander("Description"):
                        st.write(job["description"] + "…")

                # Contact RH : affiché UNIQUEMENT si réellement présent dans les données de l'offre
                if job.get("contact_linkedin"):
                    c1.markdown(f"👤 Contact RH : [{job.get('contact_name', 'Voir le profil')}]({job['contact_linkedin']})")

                if job.get("url"):
                    c2.link_button("Postuler →", job["url"])

                if st.session_state.get("student_cv_text"):
                    if c2.button("📝 Analyser mon CV", key=f"advise_{i}"):
                        with st.spinner("Génération du rapport..."):
                            report = generate_cv_improvement_report(
                                st.session_state["student_cv_text"], job["title"], job.get("description", ""),
                            )
                        st.session_state[f"report_{i}"] = report

                report = st.session_state.get(f"report_{i}")
                if report:
                    if report.get("error"):
                        st.error(f"Erreur lors de la génération du rapport : {report['error']}")
                    else:
                        if report["skills_manquantes"]:
                            st.markdown("**🎯 Compétences manquantes**")
                            st.write(", ".join(report["skills_manquantes"]))
                        if report["elements_a_retirer"]:
                            st.markdown("**✂️ À retirer ou raccourcir**")
                            for item in report["elements_a_retirer"]:
                                st.write(f"- {item}")
                        if report["conseils_generaux"]:
                            st.markdown("**💡 Conseils**")
                            for item in report["conseils_generaux"]:
                                st.write(f"- {item}")
                        if report["points_forts_a_souligner"]:
                            st.markdown("**⭐ Points forts à mettre en avant**")
                            for item in report["points_forts_a_souligner"]:
                                st.write(f"- {item}")
    elif "job_search_results" in st.session_state:
        st.info("Aucune offre trouvée pour cette recherche.")
# ---------------------------------------------------------------------------
# Routage
# ---------------------------------------------------------------------------
view = st.session_state.view
if st.session_state.role == auth.ROLE_ETUDIANT:
    render_job_search()
elif view == "domains":
    render_domains()
elif view == "jobs":
    render_jobs()
elif view == "job_detail":
    render_job_detail()
elif view == "ranking":
    render_ranking()
elif view == "candidate_detail":
    render_candidate_detail()
else:
    render_domains()
