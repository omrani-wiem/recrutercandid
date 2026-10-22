# Système multi-agent de matching CV / offres d'emploi

[![CI](https://github.com/omrani-wiem/recrutercandid/actions/workflows/ci.yml/badge.svg)](https://github.com/omrani-wiem/recrutercandid/actions/workflows/ci.yml)

Interface façon "site carrière côté RH" : on choisit un domaine (Commerce, Tech, Finance,
Opérations), on parcourt ses offres, on voit les CV qui ont candidaté, on lance l'analyse,
et on obtient un classement cliquable avec score, explication, décision et email de refus.

Architecture hybride : nœuds fonction déterministes + nœuds LLM single-shot orchestrés par LangGraph avec :
- OCR à deux niveaux : extraction texte native → fallback Qwen VLM 
- Gardrails anti prompt-injection sur tout contenu non fiable (CV, offres copiées-collées)
- Prompt engineering : Système de prompts construit avec la méthode CRAFT (Context, Role, Action, Format, Tone) + few-shot
- Scoring par pondération apprise et déterministe, enrichi par un jugement LLM
  de correspondance des compétences, avec un contexte de vocabulaire propre à chaque domaine métier
- Génération d'e-mails de refus personnalisés (recommandations concrètes basées sur le profil réel),
  avec **relecture humaine obligatoire avant tout envoi** — le graphe se met en pause et attend une
  action explicite du recruteur, il ne peut jamais deviner une approbation
- Envoi d'e-mail exposé comme outil MCP séparé (le LLM ne peut jamais envoyer directement)
- Persistance simple en JSON (offres, candidats, résultats) 
## Démo rapide (5 minutes)

**Sans clé d'API** : lancer les tests (ils simulent les appels LLM, aucun réseau nécessaire).

```bash
git clone https://github.com/omrani-wiem/recrutercandid.git
cd recrutercandid
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MISTRAL_API_KEY=placeholder python -m unittest discover -s tests -v
```

**Avec une clé Mistral** : essayer l'application complète.

```bash
cp env.example .env         # renseigner MISTRAL_API_KEY (les autres variables sont optionnelles)
python scripts/generate_sample_pdf.py   # régénère samples/example_cv.pdf si besoin
streamlit run app.py
```

1. Ouvrir <http://localhost:8501>. Au premier lancement, les mots de passe des comptes de démo
   `recruteur` et `etudiant` sont ceux de `DEMO_RECRUITER_PASSWORD` / `DEMO_STUDENT_PASSWORD`
   dans `.env`, ou, s'ils sont vides, des mots de passe aléatoires affichés **une seule fois dans
   le terminal**.
2. Se connecter en `recruteur`. Quatre offres d'exemple (Tech, Finance, Commerce, Opérations) sont déjà créées.
3. Dans la barre latérale, **Vivier de CV** : ajouter `samples/example_cv.pdf` (un CV de Data Engineer).
4. Choisir le domaine **Tech**, ouvrir l'offre *Data Engineer Senior*, puis cliquer sur
   **Analyser / relancer les candidats**.
5. Consulter le classement et le détail du candidat : score sur 20, explication, décision
   (shortlist, zone grise ou rejet) et, en cas de rejet, brouillon d'e-mail de refus **à valider
   avant tout envoi**.

L'envoi réel d'e-mails (Resend), l'OCR des CV scannés (Qwen VLM) et la recherche d'offres côté
étudiant (Adzuna) sont optionnels : sans leurs clés, le reste de l'application fonctionne.

## Structure

```
recrutercandid/
├── app.py                          # Interface Streamlit : domaines -> offres -> candidats -> classement -> détail
├── data_store.py                   # Persistance JSON (offres, candidats, résultats) + offres d'exemple
├── pipeline_runner.py              # Pont entre l'UI et les pauses/reprises du graphe LangGraph
├── state.py                        # Schéma d'état LangGraph (source de vérité du pipeline)
├── graph.py                        # Assemblage du graphe (nœuds, branchements, interruptions)
├── security/
│   └── prompt_injection_guard.py   # Hooks pre/post-LLM anti prompt-injection
├── prompts/
│   ├── system_prompts.py           # Prompts CRAFT + few-shot pour chaque nœud LLM
│   └── domain_prompts.py           # Contexte de vocabulaire métier par domaine (Commerce/Tech/Finance/Opérations)
├── nodes/
│   ├── ocr.py                      # Extraction texte native + décision de fallback
│   ├── extraction.py               # CV -> JSON structuré (LLM single-shot)
│   ├── jd_parsing.py               # Offre -> JSON structuré (LLM single-shot)
│   ├── skill_relevance.py          # Correspondance sémantique compétences (LLM, optionnel, contextualisé par domaine)
│   ├── matching_scoring.py         # Similarité + filtres + scoring pondéré (fonction pure)
│   ├── guardrails.py               # Biais / PII / cohérence (fonction + petit classifieur)
│   ├── explanation.py              # Justification ancrée dans les faits extraits (LLM)
│   ├── rejection_email.py          # Génère le brouillon de refus personnalisé (LLM), jamais auto-approuvé
│   └── human_review.py             # Points d'interruption LangGraph (interrupt())
├── mcp_servers/
│   ├── ocr_mcp_server.py           # Outil MCP : OCR Qwen VLM
│   └── email_mcp_server.py         # Outil MCP : envoi d'e-mail (seul point de sortie réel)
├── rl/
│   ├── scoring_model.py            # Bandit contextuel : poids appris à partir du feedback recruteur
│   └── feedback_store.py           # Journal des avis recruteur (👍/👎) en attente de prise en compte
└── data/                           # Créé automatiquement au premier lancement (jobs.json, candidates.json, cv_files/)
```

## Où sont stockées les données

Tout est dans `data/` (créé automatiquement, ignoré par Git) :
- `data/jobs.json` — les offres d'emploi (quelques offres d'exemple sont créées au premier
  lancement, une par domaine ; modifiable directement dans ce fichier ou via l'interface)
- `data/candidates.json` — les CV uploadés, avec pour chacun la liste des offres auxquelles
  il "candidate" (par défaut : toutes, comme demandé) et les résultats d'évaluation par offre
- `data/cv_files/` — copies des PDF uploadés

Aucune base de données : c'est volontairement de simples fichiers JSON, faciles à ouvrir et
modifier à la main. `data/` est dans `.gitignore` car `candidates.json` et `cv_files/`
contiennent des données personnelles (CV, noms, emails) — ne les committez pas. Si vous
voulez versionner uniquement vos offres, vous pouvez forcer l'ajout de `data/jobs.json`
seul (`git add -f data/jobs.json`).

## Principe de sécurité central

**Aucun nœud LLM n'a d'accès direct à un outil qui a un effet de bord réel** (envoi d'e-mail,
écriture en base, notification candidat). Ces outils vivent exclusivement dans des serveurs MCP
appelés par le graphe *après* validation humaine, jamais appelés directement par un modèle sur
la base d'un texte de CV, qui est une entrée non fiable par définition.

## Installation (Ubuntu)

### 1. Créer et activer un environnement virtuel

```bash
# Se placer dans le dossier du projet
cd recrutercandid

# Créer l'environnement virtuel (dossier .venv)
python3 -m venv .venv

# L'activer (à refaire à chaque nouvelle session de terminal)
source .venv/bin/activate

# Votre invite de commande doit maintenant afficher (.venv) au début
```

Pour désactiver l'environnement virtuel plus tard : `deactivate`.

### 2. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Dépendance requise pour l'extraction PDF native
pip install pdfplumber
```

### 3. Configurer les clés d'API

```bash
cp env.example .env
nano .env   # ou l'éditeur de votre choix, renseignez MISTRAL_API_KEY, RESEND_API_KEY, etc.
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur (par défaut sur
`http://localhost:8501`). Si vous avez démarré un serveur Qwen VLM à l'étape
précédente (option B), il doit tourner dans un terminal séparé pendant que
Streamlit tourne dans celui-ci.

### Récapitulatif (une fois tout installé)

```bash
source .venv/bin/activate
streamlit run app.py
```

## OCR (Qwen VLM) 
Quand un CV est scanné/image (extraction native de mauvaise qualité), le pipeline
bascule sur `qwen_vlm_ocr`

### Option A
```bash
# Dans .env
QWEN_VLM_URL=https://api.groq.com/openai/v1/chat/completions
QWEN_VLM_API_KEY=votre_clé_groq
QWEN_VLM_MODEL=qwen/qwen3-vl-32b-instruct
```
### Option B : auto-hébergement local avec vLLM (nécessite un GPU dédié)

Uniquement si vous disposez d'un GPU avec suffisamment de VRAM et d'espace disque libre
(vérifiez avec `df -h` avant de lancer l'installation — comptez 5 à 10 Go rien que pour
PyTorch/CUDA) :

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8001
```

## Comment le score de compatibilité est calculé

Le score final n'est **pas** décidé par un LLM qui "juge" le candidat — c'est un calcul
fixe, basé sur 4 ingrédients, chacun avec un poids d'importance :

| Ingrédient | Poids | Ce que ça mesure |
|---|---|---|
| Compétences en commun | 35% | Part des compétences demandées par l'offre que le candidat possède |
| Expérience | 25% | Années du candidat comparées au minimum demandé par l'offre |
| Ressemblance générale du profil | 30% | À quel point le texte du CV "ressemble" au texte de l'offre |
| Conditions obligatoires | 10% | Certifications ou exigences indispensables, respectées ou non |

Ces 4 notes (chacune comprise entre 0 et 1) sont combinées selon leur poids, puis
ramenées sur une échelle de 0 à 20 (voir `_scoring_model.predict` et
`_compute_score_breakdown` dans `nodes/matching_scoring.py`). **Ce calcul final reste
entièrement déterministe et auditable** : à poids et données égales, il donne toujours
le même résultat — un choix volontaire pour un outil de recrutement (traçabilité,
explicabilité en cas de contestation).

### D'où vient le premier ingrédient (compétences en commun) ?

Comparer des compétences mot pour mot est fragile : une offre qui dit "compétences en
analyse de données" au lieu de "Python, SQL" ne matchera jamais un candidat qui liste
littéralement "Python, SQL" dans ses compétences, même si c'est exactement ce qui est
demandé. Le pipeline utilise donc en priorité un nœud LLM dédié
(`nodes/skill_relevance.py`) qui juge la correspondance sémantique entre les deux
listes — **uniquement cette correspondance**, jamais le score final ni une décision
d'embauche. Ce jugement (quelle compétence a été jugée équivalente à quoi) est
conservé et visible, donc toujours vérifiable a posteriori.

Si cet appel échoue (LLM indisponible, réponse mal formée) ou n'est pas configuré, le
pipeline retombe automatiquement sur une correspondance mot-à-mot classique (avec une
petite table de synonymes), plus limitée mais 100% locale et sans dépendance réseau.
Dans l'interface, l'infobulle sur "Compétences en commun" indique quelle méthode a été
utilisée pour ce candidat.

**Ce calcul ne s'améliore pas tout seul avec l'usage.** Dans l'interface Streamlit,
chaque candidat peut recevoir un avis du recruteur (👍/👎). Ces avis sont mis de côté
(`rl/feedback_log.jsonl`) et ne sont pris en compte qu'après avoir cliqué sur
« Tenir compte de mes avis » dans le menu de gauche, ce qui ajuste légèrement les
poids ci-dessus (`offline_update` dans `rl/scoring_model.py`). Un ou deux avis isolés
ne changent presque rien au score affiché — c'est volontaire, pour éviter qu'un avis
isolé ne déforme le calcul ; il faut accumuler plusieurs dizaines d'avis pour voir une
vraie différence.

## Seuils de routage

Une fois le score calculé (sur 20), le pipeline décide automatiquement quoi faire :
- score > 14 → shortlist automatique
- score < 8 → rejet automatique (génération d'un brouillon d'email de refus, à valider)
- entre 8 et 14 → zone grise, revue humaine obligatoire (`human_review_decision`)

Ces seuils sont centralisés dans `nodes/matching_scoring.py` (`REJECT_THRESHOLD`,
`SHORTLIST_THRESHOLD`) — à ajuster selon vos besoins métier.

## Prompts par domaine (Commerce, Tech, Finance, Opérations)

`prompts/domain_prompts.py` contient un **premier jet** de contexte de vocabulaire métier
pour chaque domaine (ex: en finance, "analyse de données" évoque Excel/Power BI, pas Python ;
en tech, l'inverse). Ce contexte est injecté dans l'appel LLM de `nodes/skill_relevance.py`
pour l'aider à juger la correspondance des compétences sans se tromper de domaine.

C'est un point de départ à ajuster, pas une liste figée : le LLM garde la liberté de juger
au-delà de ce qui y est écrit. Pour l'enrichir, éditez simplement le texte associé à chaque
domaine dans `DOMAIN_SKILL_CONTEXT` — pas besoin de toucher au code des nœuds.

Pour ajouter un nouveau domaine : ajoutez sa clé dans `DOMAIN_SKILL_CONTEXT` et
`DOMAIN_LABELS` (`prompts/domain_prompts.py`) — c'est la seule source de vérité,
`data_store.py` importe `DOMAIN_LABELS` de là pour rester synchronisé automatiquement.
