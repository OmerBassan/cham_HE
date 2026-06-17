"""
Prompt templates for the VQAv2 robustness evaluation pipeline.

All three pipeline stages — Distortion, Validation, and Generation —
are purpose-built for Visual Question Answering tasks:

  • Distortion  : rephrase a natural-language question about an image
                  while keeping the expected answer identical.
  • Validation  : verify that a rephrased question is semantically
                  equivalent to the original (same answer from the image).
  • Generation  : answer a (possibly distorted) question given an image.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ============================================================================
# Distortion
# ============================================================================

_BASE_CONSTRAINT = (
    "CRITICAL: The correct answer to the question MUST remain exactly the same. "
    "Do NOT introduce new visual details, change object identities, alter quantities, "
    "colors, spatial relations, or any other attribute that would change what the image "
    "must show for the answer to be correct."
)

# μ → distortion-intensity rule, tuned for short VQA questions (avg 6-8 words).
MIU_RULES: Dict[float, str] = {
    0.0: (
        "NONE: Return the original question exactly as written. No changes allowed."
    ),
    0.1: (
        f"{_BASE_CONSTRAINT} "
        "MINIMAL LEXICAL: Replace exactly 1 content word with a direct synonym "
        "(e.g., 'What color is the car?' → 'What color is the vehicle?'). "
        "Keep word order and grammatical structure identical."
    ),
    0.2: (
        f"{_BASE_CONSTRAINT} "
        "LIGHT LEXICAL: Replace 2–3 content words with synonyms. "
        "Sentence structure must remain unchanged."
    ),
    0.3: (
        f"{_BASE_CONSTRAINT} "
        "MODERATE LEXICAL: Replace 3–4 words with synonyms or near-equivalents. "
        "Minor rephrasing of standard question phrases is allowed "
        "(e.g., 'What kind of' → 'What type of', 'How many' → 'What is the number of')."
    ),
    0.4: (
        f"{_BASE_CONSTRAINT} "
        "HEAVY LEXICAL: Extensive vocabulary substitution throughout the question. "
        "Sentence structure may shift slightly but must remain a direct, single question."
    ),
    0.5: (
        f"{_BASE_CONSTRAINT} "
        "LIGHT MIXED: Combine lexical substitutions with mild syntactic restructuring "
        "(e.g., active/passive voice swap, clause reordering). "
        "All key visual concepts (nouns, attributes, quantities) must be preserved."
    ),
    0.6: (
        f"{_BASE_CONSTRAINT} "
        "MODERATE MIXED: Significant vocabulary and structural changes. "
        "You may rephrase the question approach "
        "(e.g., 'Is the dog sitting?' → 'What is the dog's posture?') "
        "as long as the ground-truth answer does not change."
    ),
    0.7: (
        f"{_BASE_CONSTRAINT} "
        "HEAVY MIXED: Major restructuring with rare synonyms and complex syntax. "
        "The surface form should be substantially different while the answer-determining "
        "visual concept is fully preserved."
    ),
    0.8: (
        f"{_BASE_CONSTRAINT} "
        "NEAR PARAPHRASE: Extensive restructuring using different syntactic patterns. "
        "The question should be unrecognisable as a surface form of the original "
        "while pointing to exactly the same region/attribute/count in the image."
    ),
    0.9: (
        f"{_BASE_CONSTRAINT} "
        "FULL PARAPHRASE: Change every possible word and structure. "
        "The only thing preserved is the underlying visual query — "
        "what part of the image is being asked about and what the answer would be."
    ),
}

# System prompt sent once per API call (kept short to save tokens).
DISTORTION_SYSTEM_PROMPT = (
    "You are a data-augmentation expert for a Visual Question Answering (VQA) benchmark.\n"
    "Your job: rephrase questions about images without changing what the correct answer would be.\n\n"
    "HARD RULES:\n"
    "1. The rephrased question must have the same answer as the original.\n"
    "2. Do NOT add, remove, or change any visual attribute (color, count, shape, position, identity).\n"
    "3. Every rephrased question MUST end with a question mark.\n"
    "4. Your response MUST start with the exact token DISTORTED: followed by the rephrased question.\n"
    "   No preamble, no headers, no markdown — just the tagged line.\n"
    "   CORRECT: DISTORTED: Has the feline drifted off to sleep?\n"
    "   WRONG:   Lexical Changes: ...  /  Here's one distorted variation ...  /  1. ..."
)


def get_distortion_prompt(question: str, miu: float, n_distortions: int = 1) -> str:
    """
    Build the user-turn prompt for distorting a VQA question.

    Args:
        question:       The original VQA question string.
        miu:            Distortion intensity in [0.0, 1.0].
        n_distortions:  Number of distinct rephrasings to generate.

    Returns:
        Formatted prompt string ready to send as the 'user' message.
    """
    rule = MIU_RULES.get(round(miu, 1), MIU_RULES[0.5])
    lines = [
        f'Original question: "{question}"',
        "",
        f"Distortion intensity mu={miu} — Rule:",
        rule,
        "",
        "Rephrase the question following the rule above.",
        "The rephrasing must:",
        "  - End with a question mark.",
        "  - Preserve the correct answer to the original question.",
        "  - Be a complete, grammatically correct question.",
        "",
        "Reply with exactly ONE line in this format (no other text):",
        "DISTORTED: <your rephrased question here>",
    ]
    return "\n".join(lines)



# ============================================================================
# Validation
# ============================================================================

_VALIDATION_TEMPLATE = (
    "You are validating whether two VQA questions ask for the same visual information "
    "about an image, i.e., whether looking at the same image would produce "
    "the same correct answer for both.\n\n"
    "Original question:\n{original}\n\n"
    "Rephrased question:\n{distorted}\n\n"
    "Decision rules:\n"
    "  YES: the expected answer is identical for both questions.\n"
    "  NO:  the rephrasing changes what information the image must provide, "
    "or could yield a different answer.\n\n"
    "Respond with YES or NO on the first line, then optionally one brief reason."
)


def get_validation_prompt(original: str, distorted: str) -> str:
    """
    Build the validation prompt to check semantic equivalence of two VQA questions.

    Args:
        original:   The original VQA question.
        distorted:  The rephrased (distorted) VQA question.

    Returns:
        Formatted prompt string ready to send as a chat message.
    """
    return _VALIDATION_TEMPLATE.format(original=original, distorted=distorted)


# ============================================================================
# Generation (answering)
# ============================================================================

VQA_SYSTEM_PROMPT = (
    "You are an expert Visual Question Answering (VQA) system.\n"
    "You will be shown an image and asked a question about it.\n\n"
    "INSTRUCTIONS:\n"
    "- Answer ONLY based on what is visible in the image.\n"
    "- Give the shortest possible answer (typically 1-3 words).\n"
    "- Do NOT explain, justify, or add qualifiers.\n"
    "- Do NOT start with 'The answer is', 'I see', or any preamble.\n"
    "- If the question expects yes/no, reply with exactly 'yes' or 'no'.\n"
    "- If the question expects a count, reply with the numeral only (e.g., '3').\n"
    "- Match the capitalization style of a direct noun-phrase answer."
)


def get_generation_prompt(question: str, image_path: str) -> Tuple[str, str]:
    """
    Build the system and user prompts for VQA answer generation.

    Args:
        question:    The (possibly distorted) question to answer.
        image_path:  Local path to the COCO image (used by the caller to
                     attach the image; included here for traceability).

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    user_prompt = (
        f"Question: {question}\n"
        "Answer (1-3 words only):"
    )
    return VQA_SYSTEM_PROMPT, user_prompt


def build_generation_messages(question: str, image_path: str) -> List[Dict[str, str]]:
    """
    Build the full message list for a chat-completion VQA call.

    Note: image attachment is handled by the caller (vqav2.py) using
    the provider-specific multimodal API; this function provides the
    text portions only.

    Args:
        question:    The question to answer.
        image_path:  Path to the image (passed through for the caller's use).

    Returns:
        List of message dicts in OpenAI-compatible chat format.
    """
    system, user = get_generation_prompt(question, image_path)
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
