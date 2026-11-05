from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

FEATURE_ORDER = [
    "semantic_similarity",
    "skill_overlap_ratio",
    "experience_match",
    "hard_filters_passed",
]

MODEL_PATH = Path(__file__).parent / "weights.json"
LEGACY_MODEL_PATH = Path(__file__).resolve().parent.parent / "rl" / "weights.json"


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

        model_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
        if model_path.exists():
            loaded = json.loads(model_path.read_text(encoding="utf-8"))

            # On fusionne les poids chargés dans les poids par défaut
            # pour garantir la présence des nouvelles features.
            default_weights.update(loaded)

        # Sécurité : s'assurer que toutes les clés attendues existent
        for feature in FEATURE_ORDER:
            default_weights.setdefault(feature, cls().weights[feature])
        default_weights.setdefault("bias", 0.0)

        return cls(weights=default_weights)

    def save(self):
        MODEL_PATH.write_text(json.dumps(self.weights, indent=2), encoding="utf-8")


def calibrate_scoring_weights(
    feedback_log: list[dict], learning_rate: float = 0.01
) -> ScoringModel:
    """Adjust the scoring weights offline using recruiter feedback labels."""
    if not feedback_log:
        return ScoringModel.load_latest()

    model = ScoringModel.load_latest()

    feedback_values = np.array([
        e.get("feedback_label", e.get("reward")) for e in feedback_log
    ], dtype=float)

    if feedback_values.max() == feedback_values.min():
        feedback_targets = np.ones_like(feedback_values) * 0.5
    else:
        feedback_targets = (
            feedback_values - feedback_values.min()
        ) / (feedback_values.max() - feedback_values.min() + 1e-8)

    for entry, target in zip(feedback_log, feedback_targets):
        features = entry.get("features", {})
        pred = model.predict(features)
        error = target - pred

        for k in FEATURE_ORDER:
            model.weights[k] += learning_rate * error * features.get(k, 0.0)

        model.weights["bias"] += learning_rate * error

    model.save()
    return model