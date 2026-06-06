"""
judge.py — LLM judge agent for bedtime-story quality evaluation.

Two-tier evaluation:
  1. HARD BLOCKERS — binary safety gates.  ANY blocker means the story is unsafe
     and must be regenerated from scratch (not patched — you can't edit-out trauma).
  2. SOFT RUBRIC — six dimensions scored 1–5.  If the weighted overall score is
     below the pass threshold, the judge emits specific, actionable feedback and
     the storyteller gets another shot.

Why a two-tier system?
  A single score could mask a safety failure (4.5 overall but contains a death
  scene).  Blockers are a hard gate that no amount of "good writing" can override.
  This mirrors how medical-AI systems separate "do no harm" checks from quality
  metrics — appropriate for a Hippocratic AI submission.

JSON parsing strategy:
  gpt-3.5-turbo sometimes wraps JSON in markdown fences or adds preamble.
  We strip fences, try to parse, and on failure retry ONCE with a stricter prompt.
  If the retry also fails, we default to verdict="revise" so the pipeline doesn't
  surface an unjudged story to a child.
"""

import json
import logging
import re
from typing import Callable

logger = logging.getLogger("bedtime_story")

ModelFn = Callable[[str, str, float], str]

# =============================================================================
# JUDGE SYSTEM PROMPT
# =============================================================================
# Strategy: temp=0 for deterministic evaluation.  The prompt is extremely
# explicit about the JSON schema because gpt-3.5 is less reliable at structured
# output than gpt-4.  We repeat "ONLY valid JSON" multiple times and forbid
# any text outside the JSON object.
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
You are a STRICT QUALITY JUDGE for children's bedtime stories (ages 5–10).
A bedtime story is a SLEEP AID for young children.  Your job is to protect
the child from harmful content AND ensure the story is genuinely soothing.

## Tier 1 — HARD BLOCKERS (any true ⇒ story is UNSAFE)
Evaluate each blocker as true or false:
- violence: any violence, fighting, weapons, physical harm
- unresolved_fear: fear, dread, or anxiety that is NOT resolved warmly by the end
- death_grief: death, dying, loss, or heavy grief themes
- inappropriate: sexual content, profanity, drug/alcohol, hate speech, or
  content unsuitable for ages 5–10
- reading_level_too_high: vocabulary or sentence complexity clearly above age 10

## Tier 2 — SOFT RUBRIC (score each 1–5, where 5 = excellent)
- safety: overall freedom from anything that could upset a young child
- age_appropriateness: vocabulary, concepts, and themes fit ages 5–10
- bedtime_calmness: story feels like a sleep aid; energy DECREASES; ending is drowsy
- narrative_arc: clear 4-beat structure (setup → small problem → resolution → wind-down)
- read_aloud_quality: short sentences, gentle rhythm, a recurring refrain, flows well when spoken
- warmth: emotional tone is kind, reassuring, and cozy

## Scoring
- overall = average of all six rubric scores, rounded to 1 decimal
- verdict:
  - "pass"   if overall >= 4.3 AND zero blockers are true
  - "revise" otherwise

## Feedback
If verdict is "revise", provide 2–4 SPECIFIC, ACTIONABLE suggestions.
Reference exact phrases or sections.  Do NOT give vague advice.

## Output — ONLY valid JSON, no other text
{
  "blockers": {
    "violence": false,
    "unresolved_fear": false,
    "death_grief": false,
    "inappropriate": false,
    "reading_level_too_high": false
  },
  "scores": {
    "safety": 5,
    "age_appropriateness": 5,
    "bedtime_calmness": 4,
    "narrative_arc": 4,
    "read_aloud_quality": 4,
    "warmth": 5
  },
  "overall": 4.5,
  "verdict": "pass",
  "feedback": "..."
}
"""

JUDGE_TEMPERATURE = 0.0

# The pass threshold — a story must score at least this AND have zero blockers.
PASS_THRESHOLD = 4.3


def _parse_judge_json(raw: str) -> dict:
    """Parse the judge's JSON output, stripping markdown fences and trailing commas."""
    cleaned = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` wrappers.
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Remove trailing commas before ] or } (common gpt-3.5 mistake).
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    return json.loads(cleaned)


def _has_blockers(result: dict) -> bool:
    """Return True if any hard blocker is flagged."""
    blockers = result.get("blockers", {})
    return any(blockers.get(k, False) for k in [
        "violence", "unresolved_fear", "death_grief",
        "inappropriate", "reading_level_too_high",
    ])


def run_judge(model_fn: ModelFn, story: str) -> dict:
    """Evaluate a bedtime story against the safety + quality rubric.

    Returns a dict with keys: blockers, scores, overall, verdict, feedback.

    If JSON parsing fails on the first attempt, retries once with a stricter
    prompt.  If that also fails, returns a default "revise" verdict — we never
    surface an unjudged story to a child.
    """
    user_prompt = (
        f"Evaluate this bedtime story for a child aged 5–10.\n\n"
        f"--- STORY START ---\n{story}\n--- STORY END ---\n\n"
        f"Return ONLY the JSON evaluation object.  No other text."
    )

    for attempt in range(2):
        raw = model_fn(JUDGE_SYSTEM_PROMPT, user_prompt, JUDGE_TEMPERATURE)
        try:
            result = _parse_judge_json(raw)
            # Validate required keys exist.
            assert "blockers" in result
            assert "scores" in result
            assert "verdict" in result

            # Recompute overall to prevent prompt-injection / hallucinated scores.
            scores = result["scores"]
            score_values = [
                scores.get("safety", 3),
                scores.get("age_appropriateness", 3),
                scores.get("bedtime_calmness", 3),
                scores.get("narrative_arc", 3),
                scores.get("read_aloud_quality", 3),
                scores.get("warmth", 3),
            ]
            result["overall"] = round(sum(score_values) / len(score_values), 1)

            # Override verdict based on our own threshold logic.
            has_block = _has_blockers(result)
            result["has_blockers"] = has_block
            if has_block or result["overall"] < PASS_THRESHOLD:
                result["verdict"] = "revise"
            else:
                result["verdict"] = "pass"

            return result

        except (json.JSONDecodeError, KeyError, AssertionError) as e:
            logger.warning(
                "Judge JSON parse failed (attempt %d/2): %s — raw: %.200s",
                attempt + 1, e, raw,
            )
            if attempt == 0:
                # Retry with a stricter nudge appended to the user prompt.
                user_prompt += (
                    "\n\nIMPORTANT: Your previous response was not valid JSON. "
                    "Return ONLY a raw JSON object with no markdown, no explanation."
                )

    # Both attempts failed — default to "revise" so we never pass an unjudged story.
    logger.error("Judge failed to produce valid JSON after 2 attempts — defaulting to 'revise'.")
    return {
        "blockers": {k: False for k in [
            "violence", "unresolved_fear", "death_grief",
            "inappropriate", "reading_level_too_high",
        ]},
        "scores": {k: 3 for k in [
            "safety", "age_appropriateness", "bedtime_calmness",
            "narrative_arc", "read_aloud_quality", "warmth",
        ]},
        "overall": 3.0,
        "has_blockers": False,
        "verdict": "revise",
        "feedback": "Judge evaluation failed — please regenerate with extra care for safety and calmness.",
    }
