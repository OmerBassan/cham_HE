"""
Batch Processor for HumanEval-style Target Model Evaluation

This version replaces the original OpenAI Batch API + multiple-choice
evaluation with functional correctness evaluation on HumanEval-style tasks.

It expects a JSONL samples file where each line has at least:
{
    "task_id": "<HumanEval task id>",
    "completion": "<model-generated Python code>"
}

Evaluation is done by running the HumanEval unit tests and computing pass@k.
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import logging

import numpy as np
import tqdm
from dotenv import load_dotenv

# ====== CONFIGURABLE INPUT PATH ======
# Set this to the path of your HumanEval-style samples JSONL file.
# Each line should contain at least: {"task_id": "...", "completion": "..."}
# If left empty, the code will dynamically determine the path based on the project configuration.
SAMPLE_FILE_PATH: str = ""

# ====== OPTIONAL: HumanEval imports ======
try:
    from human_eval.data import HUMAN_EVAL as _DEFAULT_HUMAN_EVAL, read_problems, stream_jsonl, write_jsonl
    from human_eval.execution import check_correctness
    HAS_HUMAN_EVAL = True
except ImportError:
    _DEFAULT_HUMAN_EVAL = ""
    read_problems = None
    stream_jsonl = None
    write_jsonl = None
    check_correctness = None
    HAS_HUMAN_EVAL = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    Minimal configuration for HumanEval evaluation.

    Kept compatible with the previous interface so other modules
    (e.g., CLI, project layout) can continue to call from_project(...)
    without changes.
    """
    project_dir: Path
    model: str
    api_key: str = ""  # Not required for offline HumanEval evaluation

    # Legacy fields kept for compatibility (not used in HumanEval eval)
    max_requests_per_batch: int = 0
    completion_window: str = "24h"
    max_completion_tokens: int = 0

    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "EvaluationConfig":
        """
        Create config from project settings.

        Unlike the original version, this does NOT require an API key,
        because HumanEval evaluation runs locally on existing samples.
        """
        import yaml

        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "config.yaml"
        env_path = project_dir / ".env"

        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Optional .env load; API key is not strictly needed anymore
        if env_path.exists():
            load_dotenv(env_path)

        target_cfg = cfg.get("target_model", {})
        vendor = target_cfg.get("vendor", "openai")
        api_key = os.getenv(f"{vendor.upper()}_API_KEY", "")

        if not api_key:
            logger.info(
                "API key for vendor '%s' not found in environment. "
                "This is OK for offline HumanEval evaluation.",
                vendor,
            )

        model_name = target_cfg.get("name", "unknown-model")

        return cls(
            project_dir=project_dir,
            model=model_name,
            api_key=api_key,
        )


# ======================================================================
#  HumanEval evaluation logic (adapted from the reference implementation)
# ======================================================================


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = _DEFAULT_HUMAN_EVAL,
) -> Dict[str, float]:
    """
    Evaluates the functional correctness of generated samples and writes
    detailed per-sample results to f"{sample_file}_results.jsonl".

    Args:
        sample_file: Path to JSONL file with {"task_id", "completion"} per line.
        k:          List of k values for pass@k (typically [1]).
        n_workers:  Number of worker threads for running tests.
        timeout:    Timeout per completion in seconds.
        problem_file: Path to HumanEval problems JSONL (defaults to package constant).

    Returns:
        Dict mapping "pass@k" -> float (mean across problems).
    """
    if not HAS_HUMAN_EVAL:
        raise ImportError(
            "The 'human_eval' package is required for functional evaluation.\n"
            "Install it with: pip install human-eval"
        )

    if not problem_file:
        raise ValueError(
            "No problem_file provided and default HUMAN_EVAL path is not available."
        )

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]

            # Parse original task ID from distorted ID
            if "_d" in task_id:
                base_id = task_id.split("_d")[0]
            else:
                base_id = task_id
            
            # Normalize ID to match HumanEval format (HumanEval/0)
            # Logic: Try exact match first, then try converting human-eval_0001 -> HumanEval/1
            original_task_id = base_id
            candidates = [base_id]
            
            if base_id.lower().startswith("human-eval_"):
                try:
                    # e.g. human-eval_0001 -> 1
                    num_part = int(base_id.split("_")[-1])
                    candidates.append(f"HumanEval/{num_part}")
                except (ValueError, IndexError):
                    pass
            
            found = False
            for candidate in candidates:
                if candidate in problems:
                    original_task_id = candidate
                    found = True
                    break
            
            if not found:
                print(f"Warning: Problem {base_id} (tried {candidates}) not found for sample {task_id}")
                continue

            # Create a proxy problem with the Distorted ID so results map back correctly
            proxy_prob = problems[original_task_id].copy()
            proxy_prob["task_id"] = task_id

            args = (proxy_prob, completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) < len(problems):
             print(f"Warning: Only {len(completion_id)} unique distorted tasks evaluated.")

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort(key=lambda x: x[0])  # sort by completion_id
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {
        f"pass@{k_value}": float(estimate_pass_at_k(total, correct, k_value).mean())
        for k_value in ks
        if (total >= k_value).all()
    }

    # Finally, save the detailed results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            if results[task_id]:
                res = results[task_id].pop(0)
                sample["result"] = res[1]["result"]
                sample["passed"] = res[1]["passed"]
            else:
                sample["result"] = "skipped"
                sample["passed"] = False
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k


# ======================================================================
#  BatchProcessor wrapper (kept for compatibility with existing CLI)
# ======================================================================


class BatchProcessor:
    """
    Thin wrapper around HumanEval functional evaluation.

    The original BatchProcessor handled:
      - Creating OpenAI Batch API JSONL files
      - Submitting them to OpenAI
      - Monitoring batch jobs
      - Updating distortions CSVs

    In this HumanEval-oriented version, the core (and only) evaluation logic
    is functional correctness using the HumanEval test harness, based on
    a pre-generated samples JSONL file.

    This class is kept so that other parts of the project can still import
    and use `BatchProcessor` without changes.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

    def evaluate(
        self,
        sample_file: Optional[str] = None,
        problem_file: str = _DEFAULT_HUMAN_EVAL,
        k: List[int] = [1],
        n_workers: int = 4,
        timeout: float = 3.0,
    ) -> Dict[str, Any]:
        """
        Run HumanEval functional evaluation on the given samples file.

        Args:
            sample_file: Path to samples JSONL. If None, uses SAMPLE_FILE_PATH
                         or falls back to <project_dir>/samples.jsonl.
            problem_file: HumanEval problems file (defaults to HUMAN_EVAL).
            k: pass@k values to compute (typically [1]).
            n_workers: Number of worker threads.
            timeout: Timeout per completion.

        Returns:
            Dict with status, metrics, and output file paths.
        """
        # Decide which sample file to use
        if sample_file is None or sample_file == "":
            if SAMPLE_FILE_PATH:
                sample_file = SAMPLE_FILE_PATH
            else:
                # Fallback convention: project_dir / "samples.jsonl"
                sample_file = str(self.config.project_dir / "samples.jsonl")

        sample_path = Path(sample_file)
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Samples file not found: {sample_path}\n"
                f"Set SAMPLE_FILE_PATH at the top of this file or provide "
                f"a valid path to BatchProcessor.evaluate(...)."
            )

        print("\n" + "═" * 60)
        print("🎯 HUMAN EVAL FUNCTIONAL EVALUATION")
        print("═" * 60)
        print(f"📁 Project: {self.config.project_dir.name}")
        print(f"🤖 Model (for logging only): {self.config.model}")
        print(f"📄 Samples file: {sample_path}")
        print(f"📄 Problem file: {problem_file or '<default HUMAN_EVAL>'}")
        print(f"⚙️  pass@k: {k}")
        print(f"⚙️  Workers: {n_workers}, Timeout: {timeout}s")
        print("═" * 60 + "\n")

        metrics = evaluate_functional_correctness(
            sample_file=str(sample_path),
            k=k,
            n_workers=n_workers,
            timeout=timeout,
            problem_file=problem_file,
        )

        print("\n" + "═" * 60)
        print("📊 EVALUATION SUMMARY")
        print("═" * 60)
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        print("═" * 60 + "\n")

        return {
            "status": "complete",
            "sample_file": str(sample_path),
            "problem_file": problem_file,
            "pass_at_k": metrics,
        }


def run_evaluation(
    project_name: str,
    projects_dir: str = "Projects",
    skip_confirmation: bool = False,  # kept for compatibility, ignored
) -> Dict[str, Any]:
    """
    High-level entry point to run HumanEval functional evaluation.

    This replaces the old multi-step batch pipeline:
      - No OpenAI Batch API
      - No multiple-choice evaluation
      - No distortions CSV updates

    Instead, it:
      1. Loads project config (for consistency / logging)
      2. Chooses SAMPLE_FILE_PATH (or <project_dir>/samples.jsonl)
      3. Runs HumanEval functional correctness evaluation (pass@1 by default)
      4. Returns metrics and prints a short summary

    Args:
        project_name: Name of the project (used mainly to locate config.yaml).
        projects_dir: Base directory for projects.
        skip_confirmation: Ignored in this version (kept for API compatibility).

    Returns:
        Dict with status, pass@k metrics, and sample_file path.
    """
    config = EvaluationConfig.from_project(project_name, projects_dir)
    processor = BatchProcessor(config)

    # For now we compute only pass@1, as discussed.
    result = processor.evaluate(k=[1])

    return result


# ======================================================================
#  CLI entry point (kept compatible with previous interface)
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model on HumanEval-style generated samples"
    )
    parser.add_argument(
        "command",
        choices=["run", "create", "submit", "monitor"],
        help="Command (only 'run' is meaningful in this HumanEval version)",
    )
    parser.add_argument("--project", "-p", required=True, help="Project name")
    parser.add_argument(
        "--projects-dir", default="Projects", help="Projects directory"
    )

    args = parser.parse_args()

    config = EvaluationConfig.from_project(args.project, args.projects_dir)
    processor = BatchProcessor(config)

    if args.command == "run":
        run_evaluation(args.project, args.projects_dir)
    else:
        # The legacy commands are no-ops in this HumanEval-focused version.
        print(
            f"Command '{args.command}' is not used in the HumanEval evaluation pipeline.\n"
            f"Use:\n\n"
            f"  python batch_processor.py run --project {args.project} "
            f"--projects-dir {args.projects_dir}\n"
        )
