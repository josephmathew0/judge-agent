from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

_WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
_VOWELS = "aeiouy"

@dataclass
class TextFeatures:
    n_chars: int
    n_words: int
    avg_word_len: float
    type_token_ratio: float
    sentence_count: int
    avg_sentence_len: float
    punctuation_rate: float
    repetition_score: float
    readability_flesch: float
    has_listicles: float
    has_marketing_cta: float
    has_disclaimer_ai: float
    raw_preview: str

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _count_syllables_word(word: str) -> int:
    """
    Offline heuristic syllable counter (no NLTK/cmudict).
    Not perfect, but stable + good enough for relative readability signals.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0

    # Count vowel groups
    syllables = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel

    # Silent 'e'
    if w.endswith("e") and syllables > 1:
        syllables -= 1

    return max(syllables, 1)


def _flesch_reading_ease(text: str, words: list[str], sentences: list[str]) -> float:
    # Flesch Reading Ease:
    # 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
    n_words = len(words)
    n_sent = len(sentences)
    if n_words < 5 or n_sent == 0:
        return 0.0

    syllables = sum(_count_syllables_word(w) for w in words)
    words_per_sentence = n_words / n_sent
    syllables_per_word = syllables / n_words

    return float(206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word)


def extract_text_features(text: str) -> TextFeatures:
    text = text or ""
    n_chars = len(text)
    words = _WORD.findall(text.lower())
    n_words = len(words)

    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    ttr = (len(set(words)) / n_words) if n_words else 0.0

    sents = _split_sentences(text)
    sentence_count = len(sents)
    avg_sentence_len = float(np.mean([len(_WORD.findall(s)) for s in sents])) if sents else 0.0

    punct = sum(1 for c in text if c in ".,!?;:-")
    punctuation_rate = (punct / n_chars) if n_chars else 0.0

    bigrams = list(zip(words, words[1:]))
    if bigrams:
        uniq = len(set(bigrams))
        repetition_score = 1.0 - (uniq / len(bigrams))
    else:
        repetition_score = 0.0

    flesch = _flesch_reading_ease(text, words, sents)

    has_listicles = 1.0 if re.search(r"\b(1\.|2\.|3\.|\- |\* )", text) else 0.0
    has_marketing_cta = 1.0 if re.search(
        r"\b(like and subscribe|comment below|smash that|follow for more)\b", text.lower()
    ) else 0.0
    has_disclaimer_ai = 1.0 if re.search(
        r"\b(as an ai|i am an ai|language model)\b", text.lower()
    ) else 0.0

    return TextFeatures(
        n_chars=n_chars,
        n_words=n_words,
        avg_word_len=avg_word_len,
        type_token_ratio=ttr,
        sentence_count=sentence_count,
        avg_sentence_len=avg_sentence_len,
        punctuation_rate=punctuation_rate,
        repetition_score=repetition_score,
        readability_flesch=flesch,
        has_listicles=has_listicles,
        has_marketing_cta=has_marketing_cta,
        has_disclaimer_ai=has_disclaimer_ai,
        raw_preview=text[:300],
    )
