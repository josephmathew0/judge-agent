from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import cv2
import numpy as np
from judge_agent.utils.ffmpeg import probe


@dataclass
class VideoFeatures:
    duration_s: float
    width: int
    height: int
    sampled_frames: int
    avg_brightness: float
    motion_score: float
    sharpness_score: float
    text_overlay_likelihood: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_video_features(video_path: str, fps_sample: float = 1.0, max_frames: int = 60) -> VideoFeatures:
    meta = probe(video_path)
    duration_s = float(meta.get("duration", 0.0) or 0.0)
    width = int(meta.get("width", 0) or 0)
    height = int(meta.get("height", 0) or 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(native_fps / fps_sample), 1)

    prev_gray = None
    motions: List[float] = []
    brights: List[float] = []
    sharps: List[float] = []
    overlays: List[float] = []

    frame_idx = 0
    kept = 0
    while kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        kept += 1
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brights.append(float(np.mean(gray)))
        sharps.append(_laplacian_variance(gray))

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(float(np.mean(diff)))
        prev_gray = gray

        # crude overlay heuristic: high-contrast edges near bottom/top bands
        h, w = gray.shape
        band = gray[int(0.80*h):h, :]
        edges = cv2.Canny(band, 80, 160)
        overlays.append(float(np.mean(edges)) / 255.0)

    cap.release()

    avg_brightness = float(np.mean(brights)) if brights else 0.0
    motion_score = float(np.mean(motions)) if motions else 0.0
    sharpness_score = float(np.mean(sharps)) if sharps else 0.0
    text_overlay_likelihood = float(np.clip(np.mean(overlays) * 5.0, 0.0, 1.0)) if overlays else 0.0

    return VideoFeatures(
        duration_s=duration_s,
        width=width,
        height=height,
        sampled_frames=kept,
        avg_brightness=avg_brightness,
        motion_score=motion_score,
        sharpness_score=sharpness_score,
        text_overlay_likelihood=text_overlay_likelihood,
    )
