"""
Recherche d'offres d'emploi à l'étranger via deux sources gratuites, combinées :

- Adzuna   : nécessite un app_id + app_key gratuits (developer.adzuna.com).
             Couvre de nombreux pays (gb, us, fr, de, ca, au, nl, at, be, ch, in, ...).
- Arbeitnow: aucune clé requise. Focus Europe + remote + visa sponsorship.

Les deux APIs ne retournent que des offres actives (pas d'offres expirées) : c'est un
comportement natif de ces services, pas quelque chose qu'on doit filtrer nous-mêmes.
"""
from __future__ import annotations
import os
import urllib.parse
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

# Quelques pays supportés par Adzuna (code pays -> label pour l'UI)
ADZUNA_COUNTRIES = {
    "gb": "Royaume-Uni",
    "us": "États-Unis",
    "fr": "France",
    "de": "Allemagne",
    "ca": "Canada",
    "au": "Australie",
    "nl": "Pays-Bas",
    "at": "Autriche",
    "be": "Belgique",
    "ch": "Suisse",
    "es": "Espagne",
    "it": "Italie",
    "sg": "Singapour",
    "ae": "Émirats arabes unis",
}


def _http_get_json(url: str, timeout: int = 15) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cv-matching-system/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def search_adzuna(query: str, country: str = "gb", page: int = 1, results_per_page: int = 20) -> list[dict]:
    """Recherche via Adzuna. Renvoie une liste vide si les clés API ne sont pas configurées."""
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return []

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_APP_KEY,
        "results_per_page": results_per_page,
        "what": query,
        "content-type": "application/json",
    }
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url)
    if not data:
        return []

    jobs = []
    for item in data.get("results", []):
        jobs.append({
            "title": item.get("title", "").strip(),
            "company": (item.get("company") or {}).get("display_name", "Non renseigné"),
            "location": (item.get("location") or {}).get("display_name", ""),
            "country": ADZUNA_COUNTRIES.get(country, country.upper()),
            "url": item.get("redirect_url", ""),
            "salary_min": item.get("salary_min"),
            "salary_max": item.get("salary_max"),
            "posted_date": item.get("created", ""),
            "description": (item.get("description") or "")[:400],
            "source": "Adzuna",
        })
    return jobs


def search_arbeitnow(query: str = "", visa_sponsorship: bool | None = None) -> list[dict]:
    """Recherche via Arbeitnow (pas de clé requise). Filtre côté client sur le mot-clé."""
    url = "https://www.arbeitnow.com/api/job-board-api"
    if visa_sponsorship is not None:
        url += f"?visa_sponsorship={'true' if visa_sponsorship else 'false'}"
    data = _http_get_json(url)
    if not data:
        return []

    jobs = []
    query_lower = query.lower().strip()
    for item in data.get("data", []):
        title = item.get("title", "").strip()
        if query_lower and query_lower not in title.lower() and query_lower not in " ".join(item.get("tags", [])).lower():
            continue
        jobs.append({
            "title": title,
            "company": item.get("company_name", "Non renseigné"),
            "location": item.get("location", "") or ("Remote" if item.get("remote") else ""),
            "country": "International / Remote",
            "url": item.get("url", ""),
            "salary_min": None,
            "salary_max": None,
            "posted_date": item.get("created_at", ""),
            "description": (item.get("description") or "")[:400],
            "source": "Arbeitnow",
        })
    return jobs


def search_jobs_abroad(query: str, adzuna_country: str | None = None) -> list[dict]:
    """
    Combine les deux sources. Si adzuna_country est None, interroge uniquement Arbeitnow
    (pertinent pour une recherche "remote / international" générique).
    """
    results = []
    if adzuna_country:
        results.extend(search_adzuna(query, country=adzuna_country))
    results.extend(search_arbeitnow(query))
    return results