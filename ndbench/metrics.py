"""
Deterministic structural and surface metrics for model responses.

- Structural: list density, heading count, sentence length, readability, token
  count, step granularity, whitespace ratio.
- Surface: AI-disclaimer count, softener count, emoji count, VADER affect.

Both are pure-text, no API calls.
"""

from __future__ import annotations

import re
from statistics import mean, median

import textstat
import tiktoken
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_ENC = tiktoken.encoding_for_model("gpt-4o")
_VADER = SentimentIntensityAnalyzer()

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_NUMBERED_STEP_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.MULTILINE)
_BOLD_HEADING_RE = re.compile(r"^\s*\*\*[^*]+\*\*\s*:?\s*$", re.MULTILINE)
_CAPS_HEADING_RE = re.compile(r"^\s*[A-Z][A-Z \-]{2,}\s*:\s*$", re.MULTILINE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Hedge / AI-disclaimer phrases. Focus on patterns that signal the model is
# deflecting, self-identifying as an AI, or redirecting to a professional.
_HEDGE_PATTERNS = [
    r"\bI (?:am|'m) (?:just |only |merely )?an AI\b",
    r"\bas an AI\b",
    r"\bI (?:am|'m) not a (?:doctor|therapist|mental health|medical|licensed|qualified)\b",
    r"\bI can(?:not|'t) (?:provide|diagnose|replace|substitute|give medical)\b",
    r"\b(?:please |I recommend |I'd recommend |you should |consider )(?:consult|speaking|seeing|reaching out to|talking to) (?:a|your|with a|with your) (?:professional|therapist|doctor|counselor|mental health)\b",
    r"\bthis is not (?:medical|professional|mental health|clinical) advice\b",
    r"\bI'm not (?:able|qualified|in a position)\b",
    r"\bseek professional help\b",
]
_HEDGE_RE = re.compile("|".join(_HEDGE_PATTERNS), re.IGNORECASE)

_SOFTENER_RE = re.compile(
    r"\b(?:maybe|perhaps|might|could be|possibly|sometimes|often|generally|typically|it depends|arguably|in some cases)\b",
    re.IGNORECASE,
)

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]",
    flags=re.UNICODE,
)


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def compute_structural(text: str) -> dict:
    text = text or ""
    lines = text.split("\n") if text else [""]
    total_lines = max(1, len(lines))
    blank_lines = sum(1 for ln in lines if not ln.strip())

    sentences = _split_sentences(text)
    sent_words = [len(s.split()) for s in sentences] or [0]

    bullet_lines = len(_BULLET_RE.findall(text))
    headings = (
        len(_MD_HEADING_RE.findall(text))
        + len(_BOLD_HEADING_RE.findall(text))
        + len(_CAPS_HEADING_RE.findall(text))
    )

    # Step granularity: split on numbered markers, drop preamble, measure words per step.
    step_parts = _NUMBERED_STEP_RE.split(text)
    steps = step_parts[1:]  # drop text before first numbered marker
    step_lens = [len(s.split()) for s in steps] if steps else []

    try:
        flesch = textstat.flesch_reading_ease(text) if text.strip() else 0.0
        fkgl = textstat.flesch_kincaid_grade(text) if text.strip() else 0.0
    except Exception:
        flesch, fkgl = 0.0, 0.0

    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "token_count": count_tokens(text),
        "total_line_count": total_lines,
        "blank_line_count": blank_lines,
        "whitespace_ratio": blank_lines / total_lines,
        "sentence_count": len(sentences),
        "mean_sentence_words": mean(sent_words) if sent_words else 0.0,
        "median_sentence_words": median(sent_words) if sent_words else 0.0,
        "bullet_line_count": bullet_lines,
        "list_density": bullet_lines / total_lines,
        "heading_count": headings,
        "numbered_step_count": len(steps),
        "mean_step_words": mean(step_lens) if step_lens else 0.0,
        "flesch_reading_ease": float(flesch),
        "flesch_kincaid_grade": float(fkgl),
    }


def compute_surface(text: str) -> dict:
    text = text or ""
    v = _VADER.polarity_scores(text)
    return {
        "ai_disclaimer_count": len(_HEDGE_RE.findall(text)),
        "softener_count": len(_SOFTENER_RE.findall(text)),
        "emoji_count": len(_EMOJI_RE.findall(text)),
        "vader_compound": v["compound"],
        "vader_pos": v["pos"],
        "vader_neu": v["neu"],
        "vader_neg": v["neg"],
    }


def compute_all(text: str) -> dict:
    out = {}
    out.update(compute_structural(text))
    out.update(compute_surface(text))
    return out
