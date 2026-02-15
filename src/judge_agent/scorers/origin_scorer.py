from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


def score_origin(features: Dict[str, Any]) -> Tuple[str, float, str]:
    """
    Returns: (label, confidence, explanation)

    Explainable heuristic scoring:
      - Text: repetition + generic readability + listicle patterns + AI disclaimers
      - Video: low motion + ultra-clean sharpness + heavy overlay + short duration patterns

    Confidence is based on distance from the decision threshold (more intuitive than raw sigmoid):
      - near threshold => ~0.5–0.65
      - far from threshold => closer to 1.0
    """
    score = 0.0
    notes = []

    # Text signals
    if "text" in features:
        t = features["text"]
        rep = float(t.get("repetition_score", 0.0))
        ttr = float(t.get("type_token_ratio", 0.0))
        flesch = float(t.get("readability_flesch", 0.0))
        listicles = float(t.get("has_listicles", 0.0))
        ai_disc = float(t.get("has_disclaimer_ai", 0.0))

        score += 1.6 * rep
        score += 0.6 * listicles
        score += 2.0 * ai_disc

        # Low lexical diversity can indicate templated text
        if ttr and ttr < 0.35:
            score += 0.6
            notes.append("low lexical diversity")

        # Very "smooth" readability can correlate with generic AI copy
        if flesch and 45 <= flesch <= 80:
            score += 0.4
            notes.append("mid-high readability band")

        if rep > 0.20:
            notes.append("high phrase repetition")
        if listicles:
            notes.append("listicle/structured bullets")
        if ai_disc:
            notes.append("explicit AI disclaimer")

    # Video signals
    if "video" in features:
        v = features["video"]
        motion = float(v.get("motion_score", 0.0))
        sharp = float(v.get("sharpness_score", 0.0))
        overlay = float(v.get("text_overlay_likelihood", 0.0))
        dur = float(v.get("duration_s", 0.0))

        # Low motion + very sharp + heavy overlay can resemble templated AI short clips
        if motion < 6.0:
            score += 0.4
            notes.append("low motion")
        if sharp > 250.0:
            score += 0.4
            notes.append("very sharp frames")
        score += 0.6 * overlay

        if dur and dur < 12:
            score += 0.2
            notes.append("very short duration")

    # Map score -> label
    threshold = 1.3
    label = "ai_generated" if score >= threshold else "human_generated"

    # Confidence based on distance from threshold
    distance = abs(score - threshold)
    confidence = float(np.clip(0.5 + (distance / 3.0), 0.0, 1.0))

    # Keep confidence sensible for both sides; never below 0.5 for the chosen label
    confidence = float(np.clip(confidence, 0.5, 1.0))

    explanation = (
        f"Origin heuristic score={score:.2f}. Signals: "
        + (", ".join(notes) if notes else "no strong AI cues detected")
    )

    return label, confidence, explanation
