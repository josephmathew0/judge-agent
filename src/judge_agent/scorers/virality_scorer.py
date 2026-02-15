from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def score_virality(features: Dict[str, Any]) -> Tuple[int, str]:
    """
    Virality heuristics (0-100):
      - hooks/CTAs, short format, overlay likelihood, motion, readability, listicle structure
    """
    score = 30.0
    reasons = []

    # Text factors
    if "text" in features:
        t = features["text"]
        n_words = int(t.get("n_words", 0))
        listicles = float(t.get("has_listicles", 0.0))
        cta = float(t.get("has_marketing_cta", 0.0))
        rep = float(t.get("repetition_score", 0.0))
        raw_preview = (t.get("raw_preview", "") or "").lower()
        hook = 1.0 if any(
            phrase in raw_preview
            for phrase in ["struggling", "here are", "stop scrolling", "you're not alone", "you’re not alone"]
        ) else 0.0

        if listicles:
            score += 10
            reasons.append("structured/list format increases skimmability")
        if cta:
            score += 8
            reasons.append("explicit CTA encourages engagement")
        if n_words and n_words < 220:
            score += 6
            reasons.append("relatively short text is more shareable")
        if rep > 0.25:
            score += 3
            reasons.append("repetition can increase memorability (to a point)")
        if hook:
            score += 5
            reasons.append("strong hook increases stop-scroll potential")

    # Video factors
    if "video" in features:
        v = features["video"]
        dur = float(v.get("duration_s", 0.0))
        motion = float(v.get("motion_score", 0.0))
        overlay = float(v.get("text_overlay_likelihood", 0.0))
        bright = float(v.get("avg_brightness", 0.0))

        if dur:
            if 7 <= dur <= 35:
                score += 15
                reasons.append("short-form length fits social feeds")
            elif dur > 90:
                score -= 8
                reasons.append("longer duration reduces completion rates")

        if overlay > 0.35:
            score += 10
            reasons.append("on-screen text can improve retention without audio")

        if motion > 8.0:
            score += 6
            reasons.append("moderate motion keeps attention")
        elif motion < 3.0:
            score -= 4
            reasons.append("very low motion risks looking static")

        if bright and bright > 130:
            score += 3
            reasons.append("bright visuals tend to perform better on mobile")

    score = int(np.clip(score, 0, 100))
    explanation = "; ".join(reasons) if reasons else "No strong virality boosters detected; baseline score applied."
    return score, explanation
