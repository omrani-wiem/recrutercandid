# Guide technique — système de tri et de conseil CV

Ce document sert de support de préparation à un entretien technique. Il décrit
l'architecture et le comportement observés dans le dépôt, y compris les limites du
prototype. Les références de fichiers sont relatives à la racine du projet.

## 1. Pitch du projet

Application web Python destinée à deux profils :

- **Recruteur** : organise les offres par domaine, gère un vivier de CV, lance une
  analyse de compatibilité, consulte un classement et une explication, puis tranche
  les cas ambigus et valide séparément tout email de refus.
- **Étudiant** : recherche des offres à l'étranger avec Adzuna et Arbeitnow et peut
  demander à un LLM des conseils d'amélioration de CV pour une offre donnée.

Le projet combine une interface Streamlit, un pipeline orchestré par LangGraph, des
étapes déterministes pour le scoring et des appels LLM Mistral pour l'extraction et
certaines tâches rédactionnelles. Les données de démonstration sont stockées dans
des fichiers JSON locaux.

**Résumé oral (30 secondes)** : « J'ai construit un outil de recrutement qui extrait
les informations d'un CV, les compare à une offre et donne au recruteur un score
explicable. Le traitement est un graphe LangGraph : OCR, structuration, comparaison,
contrôles, puis orientation selon le score. Le LLM aide à comprendre et à expliquer,
mais ne décide pas seul du score final. Les cas intermédiaires passent par un humain,
et un email de refus ne peut être envoyé qu'après validation explicite. »

## 2. Architecture

```text
Streamlit (app.py)
  ├── Authentification locale (auth.py)
  ├── Données JSON et PDFs (data_store.py, data/)
  ├── Interface recruteur ── pipeline_runner.py ── LangGraph (graph.py)
  │                                              ├── PDF natif : pdfplumber
  │                                              ├── OCR de secours : client MCP -> Qwen VLM
  │                                              ├── LLM Mistral : extraction / parsing / aide
  │                                              ├── scoring et contrôles déterministes
  │                                              └── interruptions pour revue humaine
  └── Interface étudiant ── Adzuna / Arbeitnow
                         └── conseil CV par Mistral
```

### Principaux modules

| Module | Responsabilité |
|---|---|
| `app.py` | Interface Streamlit, navigation, actions recruteur/étudiant, affichage des résultats et reprise des interruptions. |
| `auth.py` | Comptes locaux, hachage des mots de passe et vérification des identifiants. |
| `data_store.py` | Offres, candidats, fichiers PDF et résultats persistés en JSON. |
| `state.py` | Modèles Pydantic des profils, offres, scores et schéma `PipelineState`. |
| `graph.py` | Déclaration des nœuds et transitions LangGraph, interruptions, OCR MCP et envoi d'email. |
| `pipeline_runner.py` | Adaptateur entre les interruptions LangGraph et les actions de l'interface. |
| `nodes/` | Étapes spécialisées du pipeline : OCR, extraction, parsing, scoring, garde-fous, explication et revue humaine. |
| `prompts/` | Instructions CRAFT, exemples few-shot et vocabulaire métier par domaine. |
| `security/prompt_injection_guard.py` | Délimitation du contenu non fiable, détection heuristique et ajout de signaux de sécurité. |
| `mcp_servers/ocr_mcp_server.py` | Serveur MCP exposant l'OCR Qwen VLM pour les CV difficiles à lire. |
| `feedback_calibration/` | Poids du score et journal des avis recruteur utilisés pour recalibrer le score hors ligne. |
| `job_search.py`, `cv_advisor.py` | Recherche d'offres externes et assistance CV côté étudiant. |

## 3. Parcours d'une analyse CV / offre

Les données initiales sont `cv_path`, `jd_text` et éventuellement le domaine
(`commerce`, `tech`, `finance` ou `operations`). L'état commun est un
`TypedDict` à champs optionnels, avec des objets Pydantic pour les données structurées.

1. **OCR natif** — `nodes/ocr.py` extrait les pages avec `pdfplumber`. `compute_confidence`
   estime la qualité selon la longueur, la proportion de caractères imprimables, la
   présence de mots-clés de sections de CV et un indicateur de texte brouillé.
2. **Fallback vision** — sous le seuil `0.55`, LangGraph appelle le serveur MCP via
   transport `stdio`. Celui-ci rasterise le PDF page par page avec PyMuPDF (`fitz`),
   puis appelle un endpoint compatible OpenAI exposant Qwen VLM. En cas d'échec,
   l'état transporte une erreur OCR explicite.
3. **Extraction du CV** — un appel Mistral transforme le texte brut en `ResumeProfile`
   (nom, email, expérience, compétences, formation, certifications, sections sources).
   Les sorties JSON sont extraites même si elles sont entourées de texte ou de balises,
   puis validées par Pydantic.
4. **Parsing de l'offre** — un appel Mistral transforme la description en
   `JobRequirements` : titre, compétences requises et souhaitées, expérience minimum,
   certifications et localisation.
5. **Correspondance des compétences** — un appel LLM optionnel évalue si les
   compétences du CV répondent aux formulations de l'offre. Le contexte métier du
   domaine choisi est ajouté pour réduire les confusions (par exemple, « analyse de
   données » n'évoque pas les mêmes outils en finance et en ingénierie data).
6. **Scoring** — `nodes/matching_scoring.py` calcule les indicateurs et le score final.
   C'est le seul endroit qui décide de la valeur numérique finale.
7. **Garde-fous et explication** — un nœud de règles signale des attributs protégés
   potentiels et vérifie les champs du score. Un appel Mistral produit ensuite une
   explication à partir des données structurées et du score, pas du texte brut du CV.
8. **Routage** — candidature retenue, rejet ou interruption de revue humaine selon les
   règles ci-dessous.
9. **Email de refus** — pour un rejet, le LLM génère uniquement un brouillon. Le graphe
   s'interrompt obligatoirement pour que le recruteur l'approuve, le modifie ou annule
   l'envoi. L'envoi est fait ensuite par le nœud séparé du graphe, avec Resend.

Le parsing de l'offre est séquentiel après l'extraction dans la version courante du
graphe. Les deux tâches ne dépendent pas fonctionnellement l'une de l'autre, mais ne
sont pas exécutées en parallèle.

## 4. Modèle de données et scoring

### Schémas principaux (`state.py`)

- `ResumeProfile` : identité, email facultatif, expérience en années, compétences,
  études, certifications et textes bruts des sections.
- `JobRequirements` : exigences structurées de l'offre.
- `MatchScore` : quatre indicateurs, score final sur 20, détails des compétences,
  filtres échoués, méthode de matching et breakdown.
- `SecurityFlag` : nœud émetteur, gravité (`info`, `warning`, `blocked`) et raison.
- `PipelineState` : entrées, résultats intermédiaires, décision, email, sécurité et
  trace du processus.

### Indicateurs

| Indicateur | Calcul et portée |
|---|---|
| Similarité de profil | Cosinus entre embeddings locaux produits par `paraphrase-multilingual-MiniLM-L12-v2` à partir des compétences/formation du CV et des compétences de l'offre. Si la bibliothèque ou le modèle est indisponible, repli journalisé vers Bag of Words ; la méthode est enregistrée dans le score. |
| Recouvrement de compétences | En priorité, jugement LLM sur les compétences requises. Si absent ou mal formé, comparaison de mots normalisés, avec quelques synonymes (par exemple `js`/`javascript`, `k8s`/`kubernetes`). Le ratio est basé sur les compétences requises. |
| Expérience | Score entre 0 et 1 : 1 si le minimum est atteint ; en dessous, pénalité progressive `sqrt(années candidat / minimum)`. Si aucun minimum n'est défini, le score vaut 1. |
| Filtres obligatoires | Actuellement, échec si une certification explicitement requise manque. L'expérience est volontairement une mesure graduelle, pas un filtre bloquant. |

Poids initiaux : similarité `0.30`, compétences `0.35`, expérience `0.25`,
filtres `0.10`, biais `0`. Le fichier `feedback_calibration/weights.json` contient les poids locaux
actuellement enregistrés et peut donc différer des valeurs initiales.

Le modèle additionne les contributions pondérées, ajoute le biais, applique une
sigmoïde recentrée, puis multiplie le résultat par 20. Le détail est exposé dans
`score_breakdown`. Le score n'est donc pas une simple somme pondérée affichée telle
quelle. À poids et entrées identiques, le résultat est déterministe.

Le remplacement du Bag of Words par des embeddings modifie la distribution de la
similarité ; les poids actuels et seuils 8/14 restent provisoires tant qu'ils ne sont
pas mesurés sur des candidatures annotées par des recruteurs. Les tests de démonstration
sur des CV/offres synthétiques contrôlent les cas typiques, mais ne constituent pas une
calibration métier représentative.

### Seuils de routage

- **Filtres obligatoires échoués** : rejet direct.
- **Score strictement supérieur à 14/20** : shortlist automatique.
- **Score strictement inférieur à 8/20** : rejet automatique.
- **De 8 à 14 inclus** : zone grise, interruption pour décision humaine.

Une décision humaine de rejet déclenche toujours une seconde interruption, distincte,
pour l'email. Un score élevé ne déclenche pas cette revue de rejet.

## 5. LLM, prompts et sécurité

Les prompts sont centralisés dans `prompts/system_prompts.py` et suivent la méthode
**CRAFT** : contexte, rôle, action, format et ton. Des exemples few-shot couvrent les
cas ordinaires, ambigus et les tentatives d'injection.

Les usages Mistral sont :

- extraction du profil CV ;
- structuration de l'offre ;
- jugement d'équivalence de compétences (facultatif, avec repli lexical) ;
- génération de l'explication du score ;
- rédaction d'un brouillon de refus ;
- conseil d'amélioration CV côté étudiant.

`cv_advisor.py` contient aussi une fonction de génération de lettre de motivation,
mais cette fonction n'est pas appelée dans le parcours Streamlit actuel.

Le garde-fou contre les prompt injections :

1. considère le CV et l'offre comme des contenus non fiables ;
2. les place entre des délimiteurs et répète la consigne après ces données ;
3. recherche des motifs suspects par expressions régulières et densité de verbes
   impératifs ;
4. vérifie certaines clés attendues dans la réponse LLM et ajoute des `SecurityFlag`.

Il s'agit d'une défense en couches, pas d'une garantie de détection exhaustive. Les
détections heuristiques produisent des signaux et ne bloquent pas systématiquement les
appels. Le texte brut du CV est destiné à l'étape d'extraction ; les étapes suivantes
utilisent principalement des données structurées. Les appels LLM ne disposent pas
d'outils capables d'envoyer un email ou de modifier directement les données métier.

Les deux interruptions reposent sur `interrupt()` et `Command(resume=...)`. Le
checkpointer utilisé est `MemorySaver` : l'état de reprise est donc en mémoire et est
perdu au redémarrage de l'application, même si score et résultat déjà enregistrés en
JSON restent consultables.

## 6. Interface, persistance et intégrations

### Interface recruteur

Navigation par domaine → offres → détail des candidats → classement → détail d'un
candidat. Le recruteur peut ajouter des offres, charger plusieurs PDF dans le vivier,
relancer les analyses en erreur, examiner les compétences et le breakdown du score,
résoudre les interruptions et enregistrer un avis 👍/👎.

Les CV ajoutés sont associés par défaut à toutes les offres existantes. Les résultats
sont affichés par offre et candidat.

### Espace étudiant

`job_search.py` combine :

- **Adzuna**, avec `ADZUNA_APP_ID` et `ADZUNA_APP_KEY`, pour les pays configurés ;
- **Arbeitnow**, sans clé, avec un filtre de recherche effectué côté client.

Le PDF envoyé par l'étudiant est extrait par OCR natif, puis le texte est gardé dans
l'état de session Streamlit pour permettre de demander un rapport de conseils sur une
offre. Ce parcours n'utilise pas le fallback OCR Qwen du pipeline recruteur.

### Persistance

- `data/jobs.json` : offres, avec offres de démonstration créées au premier chargement
  si la liste est vide.
- `data/candidates.json` : identifiants, chemin du PDF, offres associées et résultats.
- `data/cv_files/` : copies des PDF téléchargés.
- `data/users.json` : comptes et empreintes de mots de passe.
- `feedback_calibration/weights.json` : poids du score mis à jour.
- `feedback_calibration/feedback_log.jsonl` : avis recruteur en attente de prise en compte.

Il n'y a pas de base SQL : l'approche JSON simplifie le prototype, mais ne fournit pas
les garanties de concurrence, transactions, recherche ou montée en charge d'une base
de données. Les CV et journaux d'avis contiennent des données personnelles et sont
ignorés par Git.

### Authentification et configuration

`auth.py` génère un sel aléatoire par compte et stocke une empreinte PBKDF2-HMAC-SHA256
(100 000 itérations). La comparaison utilise `hmac.compare_digest`. Les comptes de
démonstration sont créés au premier chargement des utilisateurs : mot de passe via
`DEMO_RECRUITER_PASSWORD` / `DEMO_STUDENT_PASSWORD`, ou secret généré et affiché une
fois dans la console si la variable est vide.

Les variables de configuration sont décrites dans `env.example` :

- `MISTRAL_API_KEY`, `MODEL_NAME` ;
- `QWEN_VLM_URL`, `QWEN_VLM_API_KEY`, `QWEN_VLM_MODEL`,
  `QWEN_VLM_TIMEOUT_SECONDS` ;
- `RESEND_API_KEY`, `RESEND_SENDER_EMAIL`, `RESEND_MCP_URL` ;
- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY` ;
- mots de passe des comptes de démonstration.

`MISTRAL_API_KEY` est nécessaire aux fonctionnalités LLM. Recherche Adzuna, OCR VLM,
et envoi d'emails sont des intégrations optionnelles ; Arbeitnow n'exige pas de clé.
Les clés ne doivent être renseignées que dans `.env`, jamais dans le dépôt.

## 7. Recalibrage du score à partir du feedback

Quand le recruteur donne un avis sur un candidat, `feedback_calibration/feedback_store.py`
ajoute au journal JSONL un label de feedback (`1` ou `0`), les variables de score et
quelques métadonnées. Le modèle
ne se met pas à jour à chaque avis : le recruteur déclenche ensuite explicitement
`calibrate_scoring_weights()` depuis l'interface.

Cette fonction normalise les labels du lot, fait un ajustement simple par gradient
avec un taux d'apprentissage de `0.01`, puis sauvegarde `weights.json`. C'est un
ajustement simple des poids à partir des avis (feedback-based scoring calibration).
Les anciens journaux utilisant la clé `reward` restent compatibles pendant la migration.
C'est expérimental et auditable, pas un entraînement robuste de production :
il n'y a notamment pas de jeu de validation, de versionnement du modèle ou de mesure
de dérive.

## 8. Installation, lancement et tests

Exemple dans PowerShell, depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item env.example .env
# Renseigner au minimum MISTRAL_API_KEY dans .env pour les appels LLM réels.
streamlit run app.py
```

Dépendances principales : Streamlit, LangGraph, Pydantic, Mistral, MCP, pdfplumber,
PyMuPDF, NumPy, python-dotenv et requests. ReportLab est utilisé uniquement par le
script de génération du CV d'exemple.

Tests unitaires sans appels réels au LLM :

```powershell
$env:MISTRAL_API_KEY = "placeholder"
python -m unittest discover -s tests -v
```

Les tests couvrent notamment l'authentification, l'extraction JSON, le scoring et les
seuils, le fallback de matching, les interruptions/reprises du graphe, le parsing de
réponses MCP OCR, les headers Resend et les helpers UI. Les nœuds LLM sont remplacés
par des fonctions factices dans les tests d'intégration du pipeline ; ces tests ne
valident donc pas les clés, quotas, modèles hébergés ni la qualité réelle des réponses.

Le pipeline peut aussi être invoqué en ligne de commande :

```powershell
python graph.py --cv samples/example_cv.pdf --jd samples/example_offer.txt
```

Ce mode suppose un CV PDF existant et un texte d'offre disponible au chemin fourni.

## 9. Limites actuelles et pistes d'évolution

À présenter en entretien comme des arbitrages du prototype, et non comme des
fonctionnalités déjà réalisées :

- **Persistance locale** : JSON et checkpointer mémoire conviennent à une démo, pas à
  plusieurs instances ou à une reprise durable. Évolution : base relationnelle,
  stockage objet des CV et checkpointer persistant.
- **Scoring** : les embeddings multilingues améliorent la comparaison des formulations,
  mais leurs scores bruts doivent être évalués sur des candidatures annotées avant de
  considérer les poids et seuils comme calibrés pour la production. Le fallback lexical
  reste moins apte à capter le sens.
- **Feedback** : mise à jour hors ligne très simple, sans validation indépendante ni
  suivi d'équité.
- **Sécurité/PII** : heuristiques anti-injection et détection de quelques motifs
  sensibles, sans audit complet ni mécanisme d'anonymisation. Il faudrait définir
  rétention, consentement, contrôle d'accès et journalisation adaptés à la production.
- **Robustesse opérationnelle** : appels externes sensibles aux quotas, aux délais et
  à la disponibilité. Il faut surveiller les erreurs, latences, coûts, versions de
  modèle et échecs de reprise.
- **Tests** : les tests unitaires sont utiles mais les intégrations externes et la
  qualité de matching nécessitent des tests contrôlés supplémentaires.
- **Vérification avant démonstration** : dans l'état courant des fichiers, la fonction
  `_build_resend_headers()` de `graph.py` et le client Qwen de
  `mcp_servers/ocr_mcp_server.py` construisent un header `Authorization` avec la
  chaîne littérale `"******"` plutôt qu'avec la clé reçue. Les tests actuels valident
  cette valeur. Il faut corriger/vérifier ce point dans l'environnement avant de
  présenter un envoi Resend ou un endpoint Qwen authentifié comme opérationnel.

## 10. Questions d'entretien probables

**Pourquoi LangGraph plutôt qu'une fonction unique ?**  
Le graphe rend les étapes visibles, permet les branchements selon le résultat et
gère les interruptions/reprises nécessaires aux validations humaines.

**Quelle est la place du LLM dans la décision ?**  
Il extrait et structure les données, aide à rapprocher des compétences formulées
différemment, explique le résultat et rédige un brouillon. Le score numérique et les
seuils de routage restent dans du code déterministe ; le jugement de compétences est
une contribution limitée et visible, pas une décision finale.

**Comment gérez-vous un CV scanné ?**  
On tente d'abord une extraction PDF native. Un indicateur de qualité décide si l'on
appelle ensuite le serveur OCR Qwen exposé par MCP. Cette séparation permet d'isoler
le service pouvant nécessiter du GPU.

**Comment évitez-vous l'envoi automatique d'un refus ?**  
Le nœud de rédaction initialise l'approbation à `False`. Le graphe s'arrête ensuite
sur `human_review_email`; seul le payload de reprise humain peut approuver, modifier
ou annuler. Le nœud d'envoi vérifie également l'approbation.

**Pourquoi ne pas utiliser une base vectorielle ou un embedding immédiatement ?**  
Le bag-of-words local est facile à tester et à expliquer pour un MVP. Sa limite
sémantique est reconnue ; un embedding peut être évalué ensuite, sans perdre le
breakdown ni les possibilités d'audit.

**Quelles sont les limites à traiter avant la production ?**  
La persistance JSON/mémoire, l'authentification et les rôles, la protection et la
rétention des données personnelles, la robustesse des intégrations, l'évaluation de
l'équité et la validation hors échantillon du scoring.
