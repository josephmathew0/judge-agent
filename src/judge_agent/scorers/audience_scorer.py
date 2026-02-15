from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np


DEFAULT_SEGMENTS = [
    "AI/tech enthusiasts",
    "Productivity/self-improvement",
    "Creators & marketers",
    "Students/learners",
    "General social feed audience",
    "Niche hobby communities",
]

def score_audiences(features: Dict[str, Any]) -> Tuple[List[dict], str]:
    """
    Lightweight audience inference:
      - Overlay + short video => TikTok/Reels audience
      - Listicles + CTA => creators/marketing
      - High readability + short => general audience
    """
    seg_scores = {s: 0.15 for s in DEFAULT_SEGMENTS}
    whys = {s: [] for s in DEFAULT_SEGMENTS}

    if "video" in features:
        v = features["video"]
        dur = float(v.get("duration_s", 0.0))
        overlay = float(v.get("text_overlay_likelihood", 0.0))
        if dur and dur <= 40:
            seg_scores["General social feed audience"] += 0.20
            whys["General social feed audience"].append("short-form video length fits feed consumption")
        if overlay > 0.35:
            seg_scores["Creators & marketers"] += 0.18
            whys["Creators & marketers"].append("text overlays are common in creator/editing styles")
            seg_scores["Productivity/self-improvement"] += 0.08
            whys["Productivity/self-improvement"].append("overlay-driven tips format is common in advice content")

    if "text" in features:
        t = features["text"]
        listicles = float(t.get("has_listicles", 0.0))
        cta = float(t.get("has_marketing_cta", 0.0))
        readability = float(t.get("readability_flesch", 0.0))

        if listicles:
            seg_scores["Students/learners"] += 0.12
            whys["Students/learners"].append("structured bullets support quick learning")
            seg_scores["Productivity/self-improvement"] += 0.10
            whys["Productivity/self-improvement"].append("listicles map well to actionable tips")
        if cta:
            seg_scores["Creators & marketers"] += 0.15
            whys["Creators & marketers"].append("CTA language is typical of creator growth loops")
        if readability and readability > 55:
            seg_scores["General social feed audience"] += 0.10
            whys["General social feed audience"].append("high readability broadens audience")

    # Normalize and pick top 3-4
    total = sum(seg_scores.values())
    for k in seg_scores:
        seg_scores[k] = float(seg_scores[k] / total)

    top = sorted(seg_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
    result = []
    for seg, p in top:
        reason = "; ".join(whys[seg]) if whys[seg] else "broad fit based on content format signals"
        result.append({"segment": seg, "likelihood": float(np.clip(p, 0.0, 1.0)), "why": reason})

    explanation = "Audience mapping uses simple format cues (length, overlays, list structure, CTA language) rather than topic modeling."
    return result, explanation
