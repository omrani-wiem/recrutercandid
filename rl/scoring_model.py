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
            loaded = json.loads(MODEL_PATH.read_text())

            # On fusionne les poids chargés dans les poids par défaut
            # pour garantir la présence des nouvelles features.
            default_weights.update(loaded)

        # Sécurité : s'assurer que toutes les clés attendues existent
        for feature in FEATURE_ORDER:
            default_weights.setdefault(feature, cls().weights[feature])
        default_weights.setdefault("bias", 0.0)

        return cls(weights=default_weights)

    def save(self):
        MODEL_PATH.write_text(json.dumps(self.weights, indent=2))


def offline_update(feedback_log: list[dict], learning_rate: float = 0.01) -> ScoringModel:
    """
    feedback_log: [{"features": {...}, "reward": 0.0 ou 1.0}, ...]
    Descente de gradient simple sur une perte quadratique (modèle lisible et auditable).
    """
    if not feedback_log:
        return ScoringModel.load_latest()

    model = ScoringModel.load_latest()

    for entry in feedback_log:
        features = entry.get("features", {})
        target = float(entry["reward"])          # 1.0 = bon profil, 0.0 = mauvais profil
        error = target - model.predict(features)

        for k in FEATURE_ORDER:
            new_w = model.weights[k] + learning_rate * error * features.get(k, 0.0)
            model.weights[k] = max(0.0, new_w)   # pas de poids négatif : reste interprétable
        model.weights["bias"] = max(-0.2, min(0.2, model.weights["bias"] + learning_rate * error))

    model.save()
    return model