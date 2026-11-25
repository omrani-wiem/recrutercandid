from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

FEATURE_ORDER = [
    "semantic_similarity",
    "skill_overlap_ratio",
    "experience_match",
    "hard_filters_passed",
]

MODEL_PATH = Path(__file__).parent / "weights.json"


@dataclass
class ScoringModel:
    weights: dict[str, float] = field(default_factory=lambda: {
        "semantic_similarity": 0.30,
        "skill_overlap_ratio": 0.35,
        "experience_match": 0.25,
        "hard_filters_passed": 0.10,
        "bias": 0.0,
    })

    def predict(self, features: dict[str, float]) -> float:
        """
        Calcule un score dans [0, 1] via une combinaison linéaire suivie d'une sigmoïde.
        """
        x = sum(self.weights.get(k, 0.0) * features.get(k, 0.0) for k in FEATURE_ORDER)
        x += self.weights.get("bias", 0.0)

        # Sigmoïde recentrée ; plus douce qu'une coupure linéaire
        return float(1 / (1 + np.exp(-3 * (x - 0.5))))

    @classmethod
    def load_latest(cls) -> "ScoringModel":
        """
        Charge les poids depuis weights.json si présent, en complétant automatiquement
        les clés manquantes pour rester compatible avec d'anciens modèles.
        """
        default_weights = cls().weights.copy()

        if MODEL_PATH.exists():
            loaded = json.loads(MODEL_PATH.read_text(encoding="utf-8"))

            # On fusionne les poids chargés dans les poids par défaut
            # pour garantir la présence des nouvelles features.
            default_weights.update(loaded)

        # Sécurité : s'assurer que toutes les clés attendues existent
        for feature in FEATURE_ORDER:
            default_weights.setdefault(feature, cls().weights[feature])
        default_weights.setdefault("bias", 0.0)

        return cls(weights=default_weights)

    def save(self):
        MODEL_PATH.write_text(
    json.dumps(self.weights, indent=2),
    encoding="utf-8"
)


def _normalize_feature_weights(weights: dict[str, float]) -> None:
    """Poids >= 0 et somme = 1, pour garder l'échelle du score et l'explicabilité."""
    for k in FEATURE_ORDER:
        weights[k] = max(0.0, weights[k])
    total = sum(weights[k] for k in FEATURE_ORDER)
    if total <= 0:
        weights.update(ScoringModel().weights)
        return
    for k in FEATURE_ORDER:
        weights[k] /= total


def _dedupe_feedback(feedback_log: list[dict]) -> list[dict]:
    """Un seul avis par couple (candidat, offre) : le dernier. Évite les clics répétés."""
    latest: dict = {}
    for i, entry in enumerate(feedback_log):
        key = (entry.get("candidate_name"), entry.get("job_title"))
        if not all(key):
            key = ("#", i)
        latest[key] = entry
    return list(latest.values())


def offline_update(
    feedback_log: list[dict],
    learning_rate: float = 0.02,
    epochs: int = 10,
) -> ScoringModel:
    """
    Recalcule les poids à partir des poids PAR DÉFAUT et de tout l'historique d'avis.
    Idempotent : l'appeler deux fois avec le même log donne le même résultat.

    feedback_log: [{"features": {...}, "reward": 0.0 ou 1.0, ...}, ...]
    Descente de gradient simple sur une perte quadratique (modèle lisible et auditable).
    """
    model = ScoringModel()                       # toujours repartir des poids initiaux
    entries = _dedupe_feedback(feedback_log or [])
    if not entries:
        model.save()
        return model

    for _ in range(epochs):
        for entry in entries:
            features = entry.get("features", {})
            target = float(entry["reward"])      # 1.0 = bon profil, 0.0 = mauvais : pas de Min-Max
            error = target - model.predict(features)

            for k in FEATURE_ORDER:
                model.weights[k] += learning_rate * error * features.get(k, 0.0)
            _normalize_feature_weights(model.weights)

            model.weights["bias"] = max(
                -0.2, min(0.2, model.weights["bias"] + learning_rate * error)
            )

    model.save()
    return model