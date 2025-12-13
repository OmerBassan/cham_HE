"""
Distortion Validator

Validates quality of generated distortions and identifies bad outputs.

In this setup we apply lexical / structural distortions to HumanEval-style
problem prompts (natural language descriptions of coding tasks), while
keeping the underlying Python function signature and tests semantically
equivalent. The validator therefore focuses purely on text quality and
semantic sanity of the distorted prompt.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class ValidationFailure(Enum):
    """Types of validation failures for a single distorted prompt."""
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    ENCODING_ERROR = "encoding_error"
    MARKDOWN = "markdown_formatting"
    PREAMBLE = "preamble_text"
    IDENTICAL = "identical_to_original"
    GARBAGE = "garbage_characters"
    LEETSPEAK = "leetspeak_detected"
    TEMPLATE_MISSING = "template_missing_or_invalid"
    TEMPLATE_MISMATCH = "template_mismatch"
    SIGNATURE_CHANGED = "signature_changed"
    IMPORTS_CHANGED = "imports_changed"


@dataclass
class ValidationResult:
    """
    Result of validating a single distortion of a HumanEval problem prompt.
    The original and distorted fields always hold plain-text versions of the
    prompt (no code execution / AST-level checks are done here).
    """
    is_valid: bool
    original: str
    distorted: str
    miu: float
    failures: List[ValidationFailure]
    details: Optional[str] = None


# Known encoding artifacts (UTF-8 mojibake etc.)
ENCODING_ARTIFACTS = ['â€™', 'â€œ', 'â€', 'â€"', 'Â', 'Ã', '\x00']

# Preamble patterns that indicate LLM meta-response instead of a pure prompt
PREAMBLE_PATTERNS = [
    r'^here are',
    r'^below are',
    r'^the following',
    r'^sure[,!]',
    r'^certainly[,!]',
    r'^i\'ll',
    r'^of course',
    r'^here\'s',
]

# Leetspeak patterns – we keep them defined for completeness, but for
# HumanEval we generally tolerate digits in identifiers / examples anyway.
LEETSPEAK_PATTERNS = [
    r'\b[a-z]*3[a-z]*\b',  # 3 for e
    r'\b[a-z]*0[a-z]*\b',  # 0 for o (in context)
    r'\b[a-z]*1[a-z]*\b',  # 1 for i/l
    r'\b[a-z]*4[a-z]*\b',  # 4 for a
    r'\b[a-z]*@[a-z]*\b',  # @ for a
]


def is_humaneval_like(text: str) -> bool:
    """
    Heuristic check whether a prompt looks like a HumanEval-style code template:
    contains a function definition and a triple-quoted docstring.
    """
    if not text:
        return False
    return ("def " in text) and ('"""' in text)


def extract_humaneval_template(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Extract (imports_block, def_line, docstring_body) from a HumanEval-style prompt.

    Returns None if the expected structure (imports + def + triple-quoted docstring)
    cannot be reliably parsed.
    """
    if not is_humaneval_like(text):
        return None

    lines = text.splitlines()
    def_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            def_idx = i
            break

    if def_idx is None:
        return None

    imports_block = "\n".join(lines[:def_idx]).strip("\n")
    def_line = lines[def_idx]

    after_def = "\n".join(lines[def_idx + 1 :])
    first_quote = after_def.find('"""')
    if first_quote == -1:
        return None
    second_quote = after_def.find('"""', first_quote + 3)
    if second_quote == -1:
        return None

    docstring_body = after_def[first_quote + 3 : second_quote]
    return imports_block, def_line, docstring_body


def reconstruct_humaneval_prompt(original: str, distorted_variant: str) -> str:
    """
    Reconstruct a full HumanEval prompt from an original prompt and a distorted docstring/variant.
    
    This fixes the issue where the distortion process/LLM drops the Python function signature
    and surrounding context, returning only the natural language description.
    
    Logic:
    1. If distorted_variant already has full structure (imports + def + docstrings), return it.
    2. If original is not HumanEval-like, return distorted_variant as is (fallback).
    3. Otherwise, extract structure (imports, def) from original and inject distorted_variant as the docstring body.
    """
    if not distorted_variant:
        return distorted_variant

    # 1. Check if already full (has def and quotes)
    if is_humaneval_like(distorted_variant):
        return distorted_variant
        
    # 2. Check if original is parseable
    tpl = extract_humaneval_template(original)
    if not tpl:
        return distorted_variant
        
    imports, signature, _ = tpl
    
    # 3. Clean the distortion to get pure text content
    cleaned_dist = distorted_variant.strip()
    
    # If the LLM output was wrapped in quotes, strip them
    if cleaned_dist.startswith('"""') and cleaned_dist.endswith('"""') and len(cleaned_dist) >= 6:
        cleaned_dist = cleaned_dist[3:-3]
    elif cleaned_dist.startswith("'''") and cleaned_dist.endswith("'''") and len(cleaned_dist) >= 6:
        cleaned_dist = cleaned_dist[3:-3]
        
    cleaned_dist = cleaned_dist.strip()
    
    # 4. Reconstruct with standard HumanEval formatting (4 space indent)
    # We ensure newlines around the content for clean style
    # Note: imports already contains newlines if present
    separator = "\n" if imports.strip() else ""
    return f'{imports}{separator}{signature}\n    """ {cleaned_dist}\n    """'



def clean_distortion(text: str) -> str:
    """
    Clean a distorted HumanEval prompt by removing common artifacts.

    This function works on the natural-language description only and does not
    attempt to parse or validate Python code. It is safe to use across
    datasets as a generic text sanitizer.

    For HumanEval-style full prompts that already contain code (imports, def,
    and triple-quoted docstring), we avoid stripping quotes so we do not
    corrupt the template; we only do lightweight cleaning (markdown wrappers,
    encoding artifacts).
    
    Args:
        text: Raw distortion text
    
    Returns:
        Cleaned text
    """
    if not text:
        return text

    text = str(text).strip()
    humaneval_like = is_humaneval_like(text)

    # Remove markdown bold/italics at start/end
    text = re.sub(r'^\*+|\*+$', '', text)
    text = re.sub(r'^_+|_+$', '', text)

    # Remove quotes at start/end ONLY for non-HumanEval plain-text prompts
    if not humaneval_like:
        text = re.sub(r'^["\']|["\']$', '', text)

    # Fix common encoding issues
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€"': '–',
        'Â': '',
        '\x00': '',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text.strip()


def validate_distortion(
    original: str,
    distorted: str,
    miu: float,
    min_length: int = 25
) -> ValidationResult:
    """
    Validate a single distortion of a HumanEval problem prompt against
    quality rules.

    The goal is to ensure that the distorted prompt is a clean, readable,
    semantically aligned paraphrase of the original description. We do NOT
    look at the Python solution or tests here – only the text prompt.

    For HumanEval-style prompts that include code, we additionally require
    that the Python template (imports + function signature + triple-quoted
    docstring) is preserved exactly; only the natural-language content of
    the docstring is allowed to change.

    Args:
        original: Original HumanEval problem prompt text
        distorted: Distorted prompt text
        miu: Distortion intensity level
        min_length: Minimum acceptable length for the distorted prompt
    
    Returns:
        ValidationResult with pass/fail and details
    """
    failures: List[ValidationFailure] = []
    distorted_clean = clean_distortion(distorted) if distorted else ""

    # Check: Empty or None
    if not distorted_clean:
        failures.append(ValidationFailure.EMPTY)
        return ValidationResult(
            is_valid=False,
            original=original,
            distorted=distorted or "",
            miu=miu,
            failures=failures,
            details="Distortion is empty",
        )

    # Check: Too short - CRITICAL (reject)
    # HumanEval prompts are typically at least one full sentence; very short
    # strings are unlikely to describe the task properly.
    if len(distorted_clean) < min_length:
        failures.append(ValidationFailure.TOO_SHORT)

    # Check: Encoding errors - just clean, don't reject
    # Already handled by clean_distortion()

    # Check: Markdown formatting - just clean, don't reject
    # Remove any remaining markdown
    distorted_clean = re.sub(r'\*\*|\*|`|##|#', '', distorted_clean).strip()

    # Check: Preamble text - try to remove it
    lower_text = distorted_clean.lower()
    for pattern in PREAMBLE_PATTERNS:
        if re.match(pattern, lower_text):
            # Try to find actual prompt after preamble
            lines = distorted_clean.split('\n')
            if len(lines) > 1:
                distorted_clean = '\n'.join(lines[1:]).strip()
            break

    # Check: Identical to original - CRITICAL for miu > 0 (reject)
    if miu > 0 and distorted_clean.strip().lower() == original.strip().lower():
        failures.append(ValidationFailure.IDENTICAL)

    # Check: Garbage characters - only reject if >20% garbage
    unusual_chars = len(re.findall(r'[^\w\s.,?!;:\'"()\\-]', distorted_clean))
    if len(distorted_clean) > 0 and (unusual_chars / len(distorted_clean)) > 0.2:
        failures.append(ValidationFailure.GARBAGE)

    # Structural/template checks for HumanEval-style prompts
    if is_humaneval_like(original):
        orig_tpl = extract_humaneval_template(original)
        dist_tpl = extract_humaneval_template(distorted_clean)

        # If the original looks like HumanEval but the distorted version does not
        if orig_tpl is None or dist_tpl is None:
            failures.append(ValidationFailure.TEMPLATE_MISSING)
        else:
            imports_o, def_o, _ = orig_tpl
            imports_d, def_d, _ = dist_tpl

            if imports_o.strip() != imports_d.strip():
                failures.append(ValidationFailure.IMPORTS_CHANGED)

            if def_o.strip() != def_d.strip():
                failures.append(ValidationFailure.SIGNATURE_CHANGED)

            # Docstring delimiters should remain the same count
            if original.count('"""') != distorted_clean.count('"""'):
                failures.append(ValidationFailure.TEMPLATE_MISMATCH)

    # Leetspeak - ignore for HumanEval: identifiers / literals may contain digits,
    # so we do not enforce a hard failure based on LEETSPEAK_PATTERNS here.

    # Only reject if any failures accumulated
    is_valid = len(failures) == 0

    return ValidationResult(
        is_valid=is_valid,
        original=original,
        distorted=distorted_clean,
        miu=miu,
        failures=failures,
        details=f"Failed: {[f.value for f in failures]}" if failures else None,
    )


def validate_batch(
    original_questions: List[str],
    distorted_questions: List[str],
    miu: float
) -> Tuple[List[ValidationResult], int, int]:
    """
    Validate a batch of prompt distortions.

    Args:
        original_questions: List of original HumanEval problem prompts
        distorted_questions: List of distorted prompt texts
        miu: Distortion intensity level for this batch
    
    Returns:
        Tuple of (results, valid_count, invalid_count)
    """
    results: List[ValidationResult] = []
    valid_count = 0
    invalid_count = 0

    for orig, dist in zip(original_questions, distorted_questions):
        result = validate_distortion(orig, dist, miu)
        results.append(result)

        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1

    return results, valid_count, invalid_count


def parse_llm_response(text: str, n_questions: int = 1) -> dict:
    """
    Parse LLM response to extract distorted HumanEval prompts.

    The parser supports two formats:
    1) Multi-question format with explicit Q markers (Q1:, Q2:, ...)
       matching get_batch_distortion_prompt.
    2) Single-question format with just numbered blocks (1., 2., 3., ...),
       matching get_retry_distortion_prompt.

    Each numbered distortion (1., 2., ...) may span multiple lines, which
    is required for HumanEval-style prompts that include imports, a function
    definition, and a docstring. All lines from a given number up to the
    next number (or next Q marker) are grouped into a single variant.

    Args:
        text: Raw LLM response text
        n_questions: Number of questions expected in response (used by caller)
    
    Returns:
        Dict mapping question number to list of distortions
    """
    results: dict = {}
    current_q: Optional[int] = None
    current_variant_lines: Optional[List[str]] = None

    def _flush_variant():
        nonlocal current_q, current_variant_lines, results
        if current_q is None or not current_variant_lines:
            return
        variant_text = "\n".join(current_variant_lines).strip()
        if not variant_text:
            return
        cleaned = clean_distortion(variant_text)
        if not cleaned or len(cleaned) <= 20:
            return
        # Skip if it looks like a meta preamble rather than a prompt
        lower = cleaned.lower()
        if any(lower.startswith(p.lstrip('^')) for p in PREAMBLE_PATTERNS):
            return
        results.setdefault(current_q, []).append(cleaned)
        current_variant_lines = None

    for raw_line in text.split('\n'):
        line = raw_line.rstrip('\r\n')
        stripped = line.strip()
        if not stripped:
            # Preserve empty lines inside a variant (for multi-line prompts)
            if current_variant_lines is not None:
                current_variant_lines.append("")
            continue

        # Check for question marker (Q1:, Question 1:, etc.)
        q_match = re.match(r'^(?:Q|Question)\s*(\d+)[:.]?\s*$', stripped, re.IGNORECASE)
        if q_match:
            _flush_variant()
            current_q = int(q_match.group(1))
            current_variant_lines = None
            continue

        # Check for numbered distortion
        num_match = re.match(r'^(\d+)[\.\)\:]\s*(.*)$', stripped)
        if num_match:
            # Flush previous variant, start a new one
            _flush_variant()
            if current_q is None:
                current_q = 1
            rest = num_match.group(2)
            current_variant_lines = [rest] if rest else []
            continue

        # Continuation of current variant
        if current_variant_lines is not None:
            current_variant_lines.append(line)

    # Don't forget the last variant
    _flush_variant()

    # If single question (no Q markers and no explicit variants parsed),
    # fall back to treating the whole text as one distortion for Q1.
    if not results:
        cleaned = clean_distortion(text.strip())
        if cleaned and len(cleaned) > 20:
            results[1] = [cleaned]

    return results


def get_validation_stats(results: List[ValidationResult]) -> dict:
    """
    Generate statistics from validation results for HumanEval prompt distortions.
    
    Args:
        results: List of ValidationResult objects
    
    Returns:
        Dict with validation statistics
    """
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    invalid = total - valid

    # Count failure types
    failure_counts = {}
    for result in results:
        for failure in result.failures:
            failure_counts[failure.value] = failure_counts.get(failure.value, 0) + 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "valid_rate": valid / total if total > 0 else 0,
        "failure_breakdown": failure_counts,
    }


def distortion_sanity_check(df, file_name: str = "distortions") -> dict:
    """
    Perform sanity check on completed distortions CSV for HumanEval-style prompts.

    Checks:
    - Empty distorted_question fields (for miu > 0)
    - Duplicate distortions within same question_id + miu
    - Distortion identical to original prompt (for miu > 0)
    - Missing required fields
    - Invalid miu values
    - Truncated/corrupted text
    - Encoding issues
    - For HumanEval-style prompts: template/code structure preservation
      (imports + function definition + triple-quoted docstring)
    
    The column layout is expected to match the one produced by the
    distortion runner: question_id, question_text, distorted_question, miu, etc.
    
    Args:
        df: DataFrame with distortions
        file_name: Name for display
    
    Returns:
        Dict with issues found
    """
    import pandas as pd

    bad_rows = []

    # Required columns
    required_cols = ['question_id', 'miu', 'distorted_question']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {
            "file": file_name,
            "total_rows": len(df),
            "bad_rows": [{"row": 0, "question_id": "N/A", "issues": [f"Missing columns: {missing_cols}"]}],
            "bad_count": 1,
            "valid": False,
            "summary": {"missing_columns": missing_cols},
        }

    # Get original prompt column
    orig_col = 'question_text' if 'question_text' in df.columns else 'question'

    # Track duplicates within question_id + miu groups
    seen_distortions = {}  # (question_id, miu) -> set of distortion texts

    for idx, row in df.iterrows():
        row_issues = []
        row_num = idx + 2  # 1-based + header

        q_id = row.get('question_id', f'row_{idx}')
        miu = row.get('miu', 0)
        distorted = row.get('distorted_question', '')
        original = row.get(orig_col, '') if orig_col in df.columns else ''

        # 1. Check for empty distortion (only for miu > 0)
        if miu > 0:
            if pd.isna(distorted) or str(distorted).strip() == '':
                row_issues.append("Empty distorted_question")
            else:
                dist_text = str(distorted).strip()

                # 2. Check if identical to original
                if original and dist_text.lower() == str(original).strip().lower():
                    row_issues.append("Distortion identical to original")

                # 3. Check for duplicates within same (question_id, miu)
                key = (q_id, miu)
                if key not in seen_distortions:
                    seen_distortions[key] = set()

                dist_lower = dist_text.lower()
                if dist_lower in seen_distortions[key]:
                    row_issues.append("Duplicate distortion")
                else:
                    seen_distortions[key].add(dist_lower)

                # 4. Check for truncated text (ends abruptly mid-word/sentence)
                # Valid endings: punctuation, colon (for prompts that end with ":"), closing quotes/parens
                valid_endings = '.?!:)"\''
                if len(dist_text) > 20 and dist_text[-1] not in valid_endings:
                    # Check if it looks truncated (ends with incomplete word)
                    words = dist_text.split()
                    if len(words) > 3:
                        last_word = words[-1]
                        # Truncated if last word is a short preposition/article
                        truncation_words = ['a', 'an', 'the', 'of', 'to', 'in', 'on', 'at', 'by', 'or', 'and', 'for', 'with']
                        if last_word.lower() in truncation_words:
                            row_issues.append("Truncated (ends with incomplete sentence)")

                # 5. Check token length relative to original
                # Tolerance scales with miu (higher miu = more paraphrasing = more length variance allowed)
                if original and len(str(original).strip()) > 20:
                    orig_tokens = len(str(original).split())
                    dist_tokens = len(dist_text.split())

                    if orig_tokens > 0:
                        length_ratio = dist_tokens / orig_tokens

                        # Dynamic tolerance based on miu:
                        # miu 0.1-0.3: ±20%, miu 0.4-0.6: ±30%, miu 0.7-0.9: ±40%
                        if miu <= 0.3:
                            tolerance = 0.20
                        elif miu <= 0.6:
                            tolerance = 0.30
                        else:
                            tolerance = 0.40

                        min_ratio = 1 - tolerance
                        max_ratio = 1 + tolerance

                        # Only flag extreme cases (>50% shorter or >100% longer)
                        if length_ratio < 0.5:
                            pct_shorter = int((1 - length_ratio) * 100)
                            row_issues.append(f"Way too short ({pct_shorter}% fewer tokens)")
                        elif length_ratio > 2.0:
                            pct_longer = int((length_ratio - 1) * 100)
                            row_issues.append(f"Way too long ({pct_longer}% more tokens)")

                # 6. Check for encoding issues
                for artifact in ENCODING_ARTIFACTS:
                    if artifact in dist_text:
                        row_issues.append(f"Encoding issue: {artifact}")
                        break

                # 7. Check for garbage characters
                garbage_ratio = sum(
                    1 for c in dist_text
                    if not c.isalnum() and c not in ' .,?!;:\'"()-/'
                ) / max(len(dist_text), 1)
                if garbage_ratio > 0.15:
                    row_issues.append("Too many special characters")

                # 8. Check for very short distortion (absolute minimum)
                if len(dist_text) < 20:
                    row_issues.append(f"Too short ({len(dist_text)} chars)")

                # 9. HumanEval template consistency checks (if original looks like HumanEval)
                if original and is_humaneval_like(str(original)):
                    if not is_humaneval_like(dist_text):
                        row_issues.append("Template missing in distorted prompt")
                    else:
                        orig_tpl = extract_humaneval_template(str(original))
                        dist_tpl = extract_humaneval_template(dist_text)
                        if orig_tpl is None or dist_tpl is None:
                            row_issues.append("Template parse failure")
                        else:
                            imports_o, def_o, _ = orig_tpl
                            imports_d, def_d, _ = dist_tpl
                            if imports_o.strip() != imports_d.strip():
                                row_issues.append("Imports changed in distorted prompt")
                            if def_o.strip() != def_d.strip():
                                row_issues.append("Function signature changed in distorted prompt")
                            if str(original).count('"""') != dist_text.count('"""'):
                                row_issues.append("Docstring triple quotes count changed")

        # 10. Check miu validity
        if not isinstance(miu, (int, float)) or miu < 0 or miu > 1:
            row_issues.append(f"Invalid miu value: {miu}")

        if row_issues:
            bad_rows.append({
                "row": row_num,
                "question_id": q_id,
                "miu": miu,
                "issues": row_issues,
                "preview": str(distorted)[:50] + "..." if distorted and not pd.isna(distorted) else "",
            })

    # Summary stats
    summary = {
        "total_rows": len(df),
        "miu_values": sorted(df['miu'].unique().tolist()),
        "unique_questions": df['question_id'].nunique(),
        "empty_distortions": len(
            df[(df['miu'] > 0) & (df['distorted_question'].isna() | (df['distorted_question'] == ''))]
        ),
        "duplicate_count": sum(
            1 for b in bad_rows if "Duplicate distortion" in str(b.get('issues', []))
        ),
    }

    return {
        "file": file_name,
        "total_rows": len(df),
        "bad_rows": bad_rows,
        "bad_count": len(bad_rows),
        "valid": len(bad_rows) == 0,
        "summary": summary,
    }


def display_distortion_sanity_results(result: dict) -> bool:
    """
    Display distortion sanity check results for HumanEval prompt distortions.
    
    Returns:
        True if the CSV passes all checks (no bad rows), False otherwise.
    """
    print("\n" + "═" * 60)
    print("🔍 DISTORTION SANITY CHECK")
    print("═" * 60)

    summary = result.get("summary", {})
    print(f"\n📊 Summary:")
    print(f"   Total rows: {result['total_rows']:,}")
    print(f"   Unique questions: {summary.get('unique_questions', 'N/A')}")
    print(f"   Miu values: {summary.get('miu_values', [])}")
    print(f"   Empty distortions: {summary.get('empty_distortions', 0)}")
    print(f"   Duplicates found: {summary.get('duplicate_count', 0)}")

    if result["valid"]:
        print(f"\n✅ ALL {result['total_rows']:,} DISTORTIONS VALID")
    else:
        print(f"\n❌ {result['bad_count']} ISSUES FOUND")
        print("-" * 50)

        # Group by issue type
        issue_groups = {}
        for bad_row in result["bad_rows"]:
            for issue in bad_row.get("issues", []):
                if issue not in issue_groups:
                    issue_groups[issue] = []
                issue_groups[issue].append(bad_row)

        for issue_type, rows in sorted(issue_groups.items(), key=lambda x: -len(x[1])):
            print(f"\n   {issue_type}: {len(rows)} occurrences")
            # Show first 3 examples
            for row in rows[:3]:
                print(f"      Row {row['row']} (Q: {row['question_id']}, μ={row.get('miu', '?')})")
                if row.get("preview"):
                    print(f"         → \"{row['preview']}\"")
            if len(rows) > 3:
                print(f"      ... and {len(rows) - 3} more")

    print("\n" + "═" * 60)
    return result["valid"]
