"""
main.py — Orchestrator for the bedtime-story generator pipeline.

Architecture:  generate → judge → revise  (with safety-first control logic)

The pipeline is a sequence of single-purpose agents, each with its own prompt
and temperature, orchestrated by a control loop that enforces:
  1. Hard safety blockers → full regeneration (never patch unsafe content).
  2. Soft quality misses  → targeted revision with judge feedback.
  3. Best-of-N tracking   → we always return the highest-scoring draft seen.

Before submitting the assignment, describe here in a few sentences what you
would have built next if you spent 2 more hours on this project:

If I had 2 more hours I would add: (1) a fine-tuned reading-level scorer using
a lightweight model (e.g., a distilBERT classifier trained on leveled readers)
to replace the LLM-based reading-level blocker with a faster, more accurate
check, and (2) a multi-judge consensus system that runs the judge agent 3 times
per evaluation and takes the median scores, reducing single-call scoring variance
that can cause good stories to be needlessly revised or borderline stories to
slip through — paired with deterministic readability metrics (Flesch-Kincaid
grade level, average sentence length) as hard-coded guardrails that don't depend
on LLM judgment at all.
"""

import os
import sys
import json
import argparse
import logging

from openai import OpenAI
from dotenv import load_dotenv

from agents import run_intake, run_arc_planner, run_storyteller, run_revision
from judge import run_judge

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_ITERS = 3          # Maximum revision attempts before returning best draft.
POLISH_ITERS = 3       # Number of forced polish rounds when --polish is used.
PASS_THRESHOLD = 4.3   # Minimum overall score for a "pass" verdict.
MODEL = "gpt-3.5-turbo"

# ---------------------------------------------------------------------------
# Logging — every pipeline stage is logged so the flow is visible in a demo.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bedtime_story")

# ---------------------------------------------------------------------------
# OpenAI client setup (>=1.0 SDK)
# ---------------------------------------------------------------------------
load_dotenv()  # Load .env file if present.

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    logger.error("OPENAI_API_KEY not set.  Export it or add it to .env")
    sys.exit(1)

client = OpenAI(api_key=_api_key)


# =============================================================================
# call_model() — the SINGLE model entry point (per assignment rules).
# =============================================================================
# Refactored from the original skeleton to accept system/user prompts and
# temperature.  Model is hardcoded to gpt-3.5-turbo as required.
# =============================================================================

def call_model(system_prompt: str, user_prompt: str, temperature: float = 0.1,
               max_tokens: int = 3000) -> str:
    """Call gpt-3.5-turbo with separate system and user prompts.

    This is the ONLY function that touches the OpenAI API.  Every agent in
    agents.py and judge.py calls this via dependency injection (ModelFn).

    Args:
        system_prompt: Instructions defining the agent's role and output format.
        user_prompt:   The specific task or content to process.
        temperature:   Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:    Maximum response length.

    Returns:
        The model's response as a string.
    """
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# =============================================================================
# Pipeline orchestrator
# =============================================================================

def run_pipeline(user_request: str, polish: bool = False) -> str:
    """Execute the full generate→judge→revise pipeline for a story request.

    Args:
        user_request: The user's story request.
        polish: If True, force 3 additional improvement rounds after the story
                passes, to push quality beyond the minimum bar.

    Returns the final story string.
    """
    # ------------------------------------------------------------------
    # STAGE 1: Intake guardrail + classification
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 1: Intake Guardrail + Classifier")
    logger.info("=" * 60)

    try:
        intake = run_intake(call_model, user_request)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Intake parse failed (%s) — retrying once.", e)
        try:
            intake = run_intake(call_model, user_request)
        except Exception:
            # Graceful fallback: treat as safe adventure request.
            logger.warning("Intake retry failed — defaulting to safe/adventure.")
            intake = {
                "safe": True,
                "redirect_message": None,
                "category": "adventure",
                "characters": [],
                "setting": None,
                "theme": None,
                "requested_length": "medium",
            }

    logger.info("  Safe: %s", intake.get("safe"))
    logger.info("  Category: %s", intake.get("category"))
    logger.info("  Characters: %s", intake.get("characters"))
    logger.info("  Setting: %s", intake.get("setting"))

    # Safety gate: refuse unsafe requests with a gentle redirect.
    if not intake.get("safe", True):
        redirect = intake.get("redirect_message", "")
        msg = (
            f"I'd love to help, but that request isn't quite right for a "
            f"bedtime story. {redirect}\n\n"
            f"How about a story about a friendly animal, a magical garden, "
            f"or a cozy adventure instead?"
        )
        logger.info("  ⛔ Request blocked — redirecting user.")
        return msg

    # ------------------------------------------------------------------
    # STAGE 2: Arc planning
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 2: Arc Planner")
    logger.info("=" * 60)

    try:
        arc_plan = run_arc_planner(call_model, intake)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Arc planner parse failed (%s) — retrying.", e)
        try:
            arc_plan = run_arc_planner(call_model, intake)
        except Exception:
            # Minimal fallback arc so the pipeline can continue.
            logger.warning("Arc planner retry failed — using fallback arc.")
            arc_plan = {
                "beats": [
                    {"beat": 1, "label": "Gentle Setup", "summary": "Introduce characters in a cozy setting."},
                    {"beat": 2, "label": "Small Problem", "summary": "A small, gentle challenge appears."},
                    {"beat": 3, "label": "Warm Resolution", "summary": "The challenge is solved with kindness."},
                    {"beat": 4, "label": "Drowsy Wind-Down", "summary": "Everything is peaceful. Time to sleep."},
                ],
                "moral": "Kindness and courage live inside you.",
                "refrain": "All was calm, all was well.",
            }

    for beat in arc_plan.get("beats", []):
        logger.info("  Beat %s (%s): %s", beat["beat"], beat["label"], beat["summary"])
    logger.info("  Refrain: %s", arc_plan.get("refrain"))
    logger.info("  Moral: %s", arc_plan.get("moral"))

    # ------------------------------------------------------------------
    # STAGES 3–5: Story generation → Judge → Revision loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STAGE 3: Storyteller (initial draft)")
    logger.info("=" * 60)

    story = run_storyteller(call_model, arc_plan, intake)
    logger.info("  Draft generated (%d chars)", len(story))

    best_story = story
    best_score = 0.0

    for iteration in range(1, MAX_ITERS + 1):
        logger.info("=" * 60)
        logger.info("STAGE 4: Judge (iteration %d/%d)", iteration, MAX_ITERS)
        logger.info("=" * 60)

        evaluation = run_judge(call_model, story)

        # Log the evaluation details.
        logger.info("  Blockers: %s",
                     {k: v for k, v in evaluation.get("blockers", {}).items() if v})
        logger.info("  Scores: %s", evaluation.get("scores"))
        logger.info("  Overall: %.1f", evaluation.get("overall", 0))
        logger.info("  Verdict: %s", evaluation.get("verdict"))
        if evaluation.get("feedback"):
            logger.info("  Feedback: %s", evaluation["feedback"][:200])

        # Track best-of-N.
        current_score = evaluation.get("overall", 0)
        if current_score > best_score:
            best_score = current_score
            best_story = story

        # Decision logic.
        if evaluation.get("has_blockers"):
            # BLOCKER HIT — regenerate from scratch, never patch unsafe content.
            logger.warning("  BLOCKER DETECTED — regenerating from scratch.")
            story = run_storyteller(call_model, arc_plan, intake)
            continue

        if evaluation.get("verdict") == "pass":
            logger.info("  ✅ Story PASSED (overall=%.1f)", current_score)
            best_story = story
            break

        # Verdict is "revise" — send feedback to the storyteller.
        if iteration < MAX_ITERS:
            logger.info("  Revising based on judge feedback...")
            feedback = evaluation.get("feedback", "Improve calmness and read-aloud quality.")
            story = run_revision(call_model, story, feedback, arc_plan, intake)
            logger.info("  Revision %d generated (%d chars)", iteration, len(story))
        else:
            logger.info("  Max iterations reached — returning best draft (score=%.1f).", best_score)

    # ------------------------------------------------------------------
    # POLISH LOOP — forced improvement rounds even after a passing story.
    # ------------------------------------------------------------------
    if polish:
        logger.info("=" * 60)
        logger.info("POLISH MODE — %d forced improvement rounds", POLISH_ITERS)
        logger.info("=" * 60)
        story = best_story
        for p_iter in range(1, POLISH_ITERS + 1):
            logger.info("-" * 40)
            logger.info("Polish round %d/%d", p_iter, POLISH_ITERS)
            logger.info("-" * 40)

            evaluation = run_judge(call_model, story)
            current_score = evaluation.get("overall", 0)
            logger.info("  Score before polish: %.1f", current_score)

            feedback = evaluation.get("feedback", "")
            if not feedback or feedback == "...":
                feedback = (
                    "The story already passes, but push it further: "
                    "sharpen the rhythm for read-aloud flow, deepen sensory "
                    "imagery (soft textures, warm colours, gentle sounds), "
                    "and make the final wind-down even drowsier."
                )

            story = run_revision(call_model, story, feedback, arc_plan, intake)
            logger.info("  Polish revision %d generated (%d chars)", p_iter, len(story))

            # Re-judge the polished version and keep it only if it improves.
            post_eval = run_judge(call_model, story)
            post_score = post_eval.get("overall", 0)
            logger.info("  Score after polish: %.1f", post_score)

            if post_eval.get("has_blockers"):
                logger.warning("  Polish introduced a blocker — keeping previous version.")
                story = best_story
            elif post_score >= best_score:
                best_score = post_score
                best_story = story
                logger.info("  Kept polished version (%.1f >= previous best).", post_score)
            else:
                logger.info("  Polish didn't improve score (%.1f < %.1f) — keeping previous best.", post_score, best_score)
                story = best_story

        logger.info("Polish complete — final score: %.1f", best_score)

    return best_story


# =============================================================================
# Interactive loop — lets the user request changes after the story is generated.
# =============================================================================

def main():
    """Main entry point: get a request, run the pipeline, offer refinement."""
    parser = argparse.ArgumentParser(description="Bedtime Story Generator")
    parser.add_argument(
        "--polish", action="store_true",
        help="Force 3 additional improvement rounds after the story passes, "
             "pushing quality beyond the minimum bar.",
    )
    args = parser.parse_args()

    print("\n🌙  Bedtime Story Generator  🌙")
    print("=" * 40)
    if args.polish:
        print("✨ Polish mode ON — every story gets 3 extra refinement rounds.\n")
    print("Tell me what kind of bedtime story you'd like!\n")
    print("Example: A story about a girl named Alice and her best friend Bob,")
    print("         who happens to be a cat.\n")

    user_input = input("Your story request: ").strip()
    if not user_input:
        print("No request provided. Goodnight! 💤")
        return

    story = run_pipeline(user_input, polish=args.polish)

    print("\n" + "=" * 60)
    print("📖  YOUR BEDTIME STORY")
    print("=" * 60)
    print(story)
    print("=" * 60)

    # Interactive feedback loop — the user can request modifications.
    # We carry prior context (the existing story) back into the pipeline
    # by treating the modification as a new request augmented with context.
    while True:
        print("\n💬  Want any changes? (e.g., 'make it shorter', 'add a dragon')")
        print("    Type 'done' or press Enter to finish.\n")
        feedback = input("Your feedback: ").strip()

        if not feedback or feedback.lower() in ("done", "quit", "exit", "no"):
            print("\n🌟  Sweet dreams! 💤\n")
            break

        # Construct a new request that carries the existing story as context.
        augmented_request = (
            f"Original request: {user_input}\n\n"
            f"Current story:\n{story}\n\n"
            f"User's modification request: {feedback}\n\n"
            f"Please modify the story according to the user's feedback while "
            f"keeping it appropriate for ages 5–10 and suitable as a bedtime story."
        )

        story = run_pipeline(augmented_request, polish=args.polish)

        print("\n" + "=" * 60)
        print("📖  REVISED BEDTIME STORY")
        print("=" * 60)
        print(story)
        print("=" * 60)


if __name__ == "__main__":
    main()
