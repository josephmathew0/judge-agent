from __future__ import annotations
from typing import Optional, Dict, Any

from judge_agent.schemas import JudgeOutput, OriginPrediction, AudienceSegment
from judge_agent.scorers.origin_scorer import score_origin
from judge_agent.scorers.virality_scorer import score_virality
from judge_agent.scorers.audience_scorer import score_audiences


def judge(
    text: Optional[str] = None,
    video_path: Optional[str] = None,
    transcript_path: Optional[str] = None,
    fps_sample: float = 1.0,
    max_frames: int = 60,
    include_debug: bool = False,
) -> JudgeOutput:
    from judge_agent.feature_extractors.text_features import extract_text_features

    features: Dict[str, Any] = {}

    if text is not None:
        tf = extract_text_features(text)
        features["text"] = tf.as_dict()

    if video_path is not None:
        from judge_agent.feature_extractors.video_features import extract_video_features
        from judge_agent.feature_extractors.audio_features import extract_audio_features

        vf = extract_video_features(video_path, fps_sample=fps_sample, max_frames=max_frames)
        features["video"] = vf.as_dict()
        af = extract_audio_features(video_path, transcript_path=transcript_path)
        features["audio"] = af.as_dict()

        # If transcript exists, also run text features on it as additional signal
        if transcript_path:
            try:
                transcript = open(transcript_path, "r", encoding="utf-8", errors="ignore").read()
                features["transcript_text"] = extract_text_features(transcript).as_dict()
            except Exception:
                pass

    # Combine transcript text into scoring by treating it as text if main text missing
    scoring_features = dict(features)
    if "text" not in scoring_features and "transcript_text" in scoring_features:
        scoring_features["text"] = scoring_features["transcript_text"]

    origin_label, origin_conf, origin_expl = score_origin(scoring_features)
    virality, virality_expl = score_virality(scoring_features)
    audiences, audience_expl = score_audiences(scoring_features)

    out = JudgeOutput(
        origin_prediction=OriginPrediction(label=origin_label, confidence=origin_conf),
        virality_score=virality,
        distribution_analysis=[AudienceSegment(**a) for a in audiences],
        explanations={
            "origin_prediction": origin_expl,
            "virality_score": virality_expl,
            "distribution_analysis": audience_expl,
        },
        debug=features if include_debug else None,
    )
    return out
