"""
agents.py — Single-purpose agent prompts and functions for the bedtime-story pipeline.

Each agent is a thin wrapper: a carefully tuned system prompt, a user-prompt
formatter, and a temperature setting.  The actual LLM call is always delegated
to call_model() in main.py (the single model entry point).

Design note — why separate agents instead of one mega-prompt?
  1. Each prompt can be tuned to a narrow task → higher reliability from gpt-3.5.
  2. Temperature per stage: classifiers need determinism (0), storytelling needs
     creativity (0.8), the judge needs strict JSON (0).
  3. Failures are isolated: a bad classification doesn't poison the story draft.
"""

import json
import logging
import re
from typing import Callable

logger = logging.getLogger("bedtime_story")

# ---------------------------------------------------------------------------
# Type alias for the model-calling function injected from main.py.
# Signature: (system_prompt, user_prompt, temperature) -> str
# ---------------------------------------------------------------------------
ModelFn = Callable[[str, str, float], str]


def _clean_json(raw: str) -> str:
    """Strip markdown fences and trailing commas from LLM-generated JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)
    # Remove trailing commas before ] or } (common gpt-3.5 mistake).
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned

# ========================== VALID CATEGORIES =================================
VALID_CATEGORIES = {
    "adventure",
    "animal_friend",
    "magical",
    "friendship",
    "silly_funny",
    "overcoming_fear",
}

# =============================================================================
# AGENT 1 — INTAKE GUARDRAIL + CLASSIFIER
# =============================================================================
# Strategy: a single call that does two jobs (saves a round-trip on gpt-3.5).
# temp=0 because classification and safety gating must be deterministic.
# We ask for strict JSON so downstream parsing is reliable.
# The "overcoming_fear" category is explicitly surfaced so that requests about
# nighttime fears get a *normalizing* story rather than a scary one.
# =============================================================================

INTAKE_SYSTEM_PROMPT = """\
You are a SAFETY GATE and REQUEST CLASSIFIER for a children's bedtime-story app.
Your audience is children ages 5–10 and their parents.

## Your two jobs

### Job 1 — Safety check the REQUEST itself
Reject or redirect if the request contains:
- Violence, weapons, gore, death, or grievous harm
- Sexual or romantic content
- Profanity, slurs, hate speech, or discriminatory themes
- Horror, intense fear without resolution, or nightmares
- Drug/alcohol references
- Content clearly meant for adults, not children

If unsafe, set "safe" to false and provide a gentle redirect message.

### Job 2 — Classify into exactly ONE category
Categories (pick the BEST fit):
- adventure        — exploration, quests, journeys
- animal_friend    — stories centered on animal characters/companions
- magical          — magic, enchanted worlds, fairy-tale elements
- friendship       — bonds between characters, teamwork, kindness
- silly_funny      — humor, absurdity, playful nonsense
- overcoming_fear  — the child is scared of something (dark, monsters, storms);
                     generate a gentle, normalizing story where the fear is
                     resolved with warmth and reassurance

### Job 3 — Extract entities
Pull out any character names, settings, themes, and requested length the user
mentioned.  If not mentioned, leave them as null.

## Output format — ONLY valid JSON, nothing else
{
  "safe": true/false,
  "redirect_message": "..." or null,
  "category": "one_of_the_six_categories",
  "characters": ["name1", ...] or [],
  "setting": "..." or null,
  "theme": "..." or null,
  "requested_length": "short" | "medium" | "long" | null
}
"""

INTAKE_TEMPERATURE = 0.0


def run_intake(model_fn: ModelFn, user_request: str) -> dict:
    """Safety-check and classify a user's story request.

    Returns a parsed dict with keys: safe, redirect_message, category,
    characters, setting, theme, requested_length.
    Raises ValueError on unparseable output (caller should retry or default).
    """
    raw = model_fn(INTAKE_SYSTEM_PROMPT, user_request, INTAKE_TEMPERATURE)

    result = json.loads(_clean_json(raw))

    # Normalize category to a known value; default to "adventure" if hallucinated.
    cat = (result.get("category") or "adventure").lower().strip()
    if cat not in VALID_CATEGORIES:
        logger.warning("Unknown category '%s' from classifier — defaulting to 'adventure'", cat)
        cat = "adventure"
    result["category"] = cat

    return result


# =============================================================================
# AGENT 2 — ARC PLANNER
# =============================================================================
# Strategy: produce a SHORT outline following the "descending bedtime arc" —
# emotional energy must DECREASE in the final third.  This is the structural
# backbone that makes the story a sleep aid rather than a thriller.
# temp=0.4 for mild creativity while staying structured.
# The planner also proposes a recurring soothing refrain (e.g., "the stars
# hummed softly…") — repetition is a known calming mechanism for children.
# =============================================================================

ARC_PLANNER_SYSTEM_PROMPT = """\
You are a STORY ARC PLANNER for children's bedtime stories (ages 5–10).
A bedtime story is a SLEEP AID read aloud by a tired parent.  The emotional
energy must DECREASE across the story, ending in drowsy calm.

## The 4-beat "descending bedtime arc"
1. GENTLE SETUP — introduce characters and a cozy setting with sensory detail.
2. SMALL PROBLEM — a low-stakes, age-appropriate challenge (never scary).
3. WARM RESOLUTION — the problem is solved through kindness, cleverness, or
   friendship. The emotional peak is MILD — no high-tension climax.
4. DROWSY WIND-DOWN — the world grows quiet; characters yawn, settle in, close
   eyes. End with a line that bridges to the child's own sleep.

## Your output — ONLY valid JSON
{
  "beats": [
    {"beat": 1, "label": "Gentle Setup", "summary": "..."},
    {"beat": 2, "label": "Small Problem", "summary": "..."},
    {"beat": 3, "label": "Warm Resolution", "summary": "..."},
    {"beat": 4, "label": "Drowsy Wind-Down", "summary": "..."}
  ],
  "moral": "a short warm takeaway (1 sentence)",
  "refrain": "a soothing repeated phrase the child can anticipate (≤10 words)"
}

## Rules
- Tailor the arc to the CATEGORY and any extracted entities provided.
- For "overcoming_fear": beat 2 should gently name the fear; beat 3 should
  normalize it and resolve it with comfort, NOT with the fear being "defeated."
- The refrain should be calm and rhythmic — think lullaby, not chant.
- Keep each beat summary to 1–2 sentences.
"""

ARC_PLANNER_TEMPERATURE = 0.4


def run_arc_planner(model_fn: ModelFn, intake_result: dict) -> dict:
    """Generate a 4-beat descending arc tailored to the classified request.

    Returns a dict with keys: beats, moral, refrain.
    """
    # Build a user prompt that passes the classification context to the planner.
    user_prompt = (
        f"Category: {intake_result['category']}\n"
        f"Characters: {intake_result.get('characters', [])}\n"
        f"Setting: {intake_result.get('setting', 'not specified')}\n"
        f"Theme: {intake_result.get('theme', 'not specified')}\n"
        f"Requested length: {intake_result.get('requested_length', 'medium')}\n"
        f"\nPlan the 4-beat descending bedtime arc."
    )
    raw = model_fn(ARC_PLANNER_SYSTEM_PROMPT, user_prompt, ARC_PLANNER_TEMPERATURE)

    return json.loads(_clean_json(raw))


# =============================================================================
# AGENT 3 — STORYTELLER
# =============================================================================
# Strategy: the creative writer.  temp ~0.8 for vivid but constrained prose.
# Hard constraints enforce age-appropriateness at the generation level (belt)
# so the judge (suspenders) catches only edge-case drift.
# The refrain from the planner is injected so the storyteller weaves it in
# 2–3 times — repetition signals "this is predictable and safe" to the child.
# =============================================================================

STORYTELLER_SYSTEM_PROMPT = """\
You are a BEDTIME STORYTELLER for children ages 5–10.
A bedtime story is a SLEEP AID.  Your goal is to make the child feel safe,
warm, and drowsy by the end.

## Hard constraints (never break these)
- Sentences: ≤12 words each. Shorter is better.
- Vocabulary: age 5–10. No complex or scary words.
- Imagery: sensory but CALM — soft textures, warm colors, gentle sounds.
- NO violence, death, intense fear, peril, or anything that could cause
  nightmares.
- Weave the provided REFRAIN into the story 2–3 times at natural pauses.
- End with a SOOTHING FINAL LINE that bridges to the child's own sleep
  (e.g., "…and she closed her eyes, just like you will, too.").

## Tone
- Warm, gentle, rhythmic — like a parent's quiet voice.
- The emotional energy must DECREASE in the second half.
- The final paragraph should feel like a whisper.

## Structure
Follow the 4-beat arc provided.  Each beat should be 2–4 short paragraphs.
The overall story should be ~300–500 words (adjustable by requested length).
"""

STORYTELLER_TEMPERATURE = 0.8


def run_storyteller(model_fn: ModelFn, arc_plan: dict, intake_result: dict) -> str:
    """Draft a full bedtime story from the arc plan.

    Returns the story as a plain-text string.
    """
    # Determine target length from the user's request.
    length_map = {"short": "200–300", "medium": "300–500", "long": "500–700"}
    length = intake_result.get("requested_length") or "medium"
    word_range = length_map.get(length, "300–500")

    beats_text = "\n".join(
        f"Beat {b['beat']} ({b['label']}): {b['summary']}"
        for b in arc_plan["beats"]
    )
    user_prompt = (
        f"## Arc plan\n{beats_text}\n\n"
        f"## Moral / warm takeaway\n{arc_plan['moral']}\n\n"
        f"## Recurring refrain (use 2–3 times)\n{arc_plan['refrain']}\n\n"
        f"## Characters\n{intake_result.get('characters', ['(your choice)'])}\n"
        f"## Setting\n{intake_result.get('setting', '(your choice)')}\n\n"
        f"## Target length\n{word_range} words\n\n"
        f"Write the bedtime story now.  Follow the arc exactly.  "
        f"Remember: sentences ≤12 words, calm imagery, and end with a drowsy "
        f"bridge to the child's own sleep."
    )
    return model_fn(STORYTELLER_SYSTEM_PROMPT, user_prompt, STORYTELLER_TEMPERATURE)


# =============================================================================
# AGENT 3b — REVISION STORYTELLER
# =============================================================================
# Strategy: when the judge says "revise", we send the SAME storyteller prompt
# but APPEND the judge's specific feedback.  This lets the model keep its
# creative momentum while addressing targeted issues.
# We lower temperature slightly (0.7) to reduce drift during revision.
# =============================================================================

REVISION_TEMPERATURE = 0.7


def run_revision(model_fn: ModelFn, current_story: str, feedback: str,
                 arc_plan: dict, intake_result: dict) -> str:
    """Revise an existing story based on judge feedback.

    Returns the revised story as plain text.
    """
    user_prompt = (
        f"## Current story draft\n{current_story}\n\n"
        f"## Judge feedback (address ALL of these)\n{feedback}\n\n"
        f"## Recurring refrain (keep using it 2–3 times)\n{arc_plan['refrain']}\n\n"
        f"## Rules reminder\n"
        f"- Sentences ≤12 words, vocabulary ages 5–10.\n"
        f"- Calm, warm, drowsy tone.  Emotional energy DECREASES.\n"
        f"- End with a soothing line bridging to the child's own sleep.\n"
        f"- NO violence, death, fear without resolution.\n\n"
        f"Rewrite the story, incorporating the feedback.  "
        f"Output ONLY the revised story."
    )
    return model_fn(STORYTELLER_SYSTEM_PROMPT, user_prompt, REVISION_TEMPERATURE)
