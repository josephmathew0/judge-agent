from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

from judge_agent.utils.ffmpeg import extract_audio

@dataclass
class AudioFeatures:
    has_audio: bool
    transcript_present: bool
    transcript_len_words: int

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__


def extract_audio_features(video_path: str, transcript_path: Optional[str] = None) -> AudioFeatures:
    # We don't do ASR by default (keeps it lightweight). If user provides transcript file, use it.
    transcript_present = False
    transcript_len_words = 0

    if transcript_path and Path(transcript_path).exists():
        transcript_present = True
        txt = Path(transcript_path).read_text(encoding="utf-8", errors="ignore")
        transcript_len_words = len(txt.split())

    # We'll assume audio exists if ffmpeg can extract it; if ffmpeg not installed, mark unknown as False.
    has_audio = False
    tmp = Path(".tmp_audio.wav")
    try:
        extract_audio(video_path, str(tmp))
        has_audio = tmp.exists() and tmp.stat().st_size > 0
    except Exception:
        has_audio = False
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    return AudioFeatures(
        has_audio=has_audio,
        transcript_present=transcript_present,
        transcript_len_words=transcript_len_words,
    )
