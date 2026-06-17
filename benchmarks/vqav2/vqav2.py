"""
VQAv2 Benchmark — Robustness Evaluation Pipeline.

Evaluates how sensitive a Vision-Language Model's accuracy is to
natural-language rephrasing of VQA questions.

Pipeline:
  1. Load tasks from the sampled VQAv2 JSON file.
  2. Distort questions via Mistral (text-only; image not needed for rephrasing).
  3. Validate that each distortion preserves the original expected answer.
  4. Generate free-text answers for both original and distorted questions
     using the configured vision-language model (see config.py).
  5. Score answers with VQA soft accuracy against the 10 annotator answers.
  6. Report per-task and aggregate metrics (VQA accuracy, exact match, μ).

Provider routing:
  Distortion & validation  → Mistral (text-only).
  Answer generation        → provider declared in ALLOWED_MODELS (config.py);
                             must support vision input.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from chameleon.core.base import BaseBenchmark
from benchmarks.base.types import (
    BenchmarkMetrics,
    BenchmarkValidationResult,
    DistortionPrompt,
    EvalResult,
    Task,
)
from benchmarks.vqav2.config import ALLOWED_MODELS
from benchmarks.vqav2.engine.data import load_entries, normalize_answers
from benchmarks.vqav2.engine.execution import evaluate_answer, vqa_score
from benchmarks.vqav2.engine.evaluate_functional_correctness import compute_vqa_metrics
from benchmarks.vqav2.prompts import (
    DISTORTION_SYSTEM_PROMPT,
    VQA_SYSTEM_PROMPT,
    build_generation_messages,
    get_distortion_prompt,
    get_validation_prompt,
)
from chameleon.distortion.engine import DistortionEngine

load_dotenv()

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None

try:
    from google import genai as _google_genai
    from google.genai import types as _google_types
except ImportError:
    _google_genai = None
    _google_types = None


def _get_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")


class VQAV2Benchmark(BaseBenchmark):
    """
    VQAv2 robustness benchmark pipeline.

    Pipeline steps:
    1. Load tasks from JSON
    2. Distort questions using Mistral
    3. Validate distortions preserve semantic meaning (Mistral)
    4. Generate answers for original and distorted questions (provider from ALLOWED_MODELS)
    5. Evaluate answers with VQA soft accuracy
    6. Calculate metrics
    """

    # VQAv2 generation needs images + provider-specific vision calls that the
    # generic text stages can't perform, so the workflow delegates the whole run
    # to run_full_pipeline() below.
    runs_own_pipeline: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("VQAV2Benchmark")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _call_with_retry(self, fn, *args, max_retries: int = 5, base_delay: int = 15, **kwargs):
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if "429" in str(e) or "rate_limited" in str(e).lower() or "RESOURCE_EXHAUSTED" in str(e):
                    delay = base_delay
                    self.logger.warning(f"Rate limited — retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise
        return fn(*args, **kwargs)

    # =========================================================================
    # AbstractBenchmark required interface
    # =========================================================================

    def load_data(self, data_path: str) -> List[Task]:
        raw_tasks = self._load_raw_tasks(data_path)
        return [Task(task_id=t["question_id"], data=t) for t in raw_tasks]

    def get_field_to_distort(self) -> str:
        return "question"

    def get_distortion_prompt(self, task: Task, miu: float, n_distortions: int) -> DistortionPrompt:
        user_prompt = get_distortion_prompt(task.data["question"], miu, n_distortions)
        return DistortionPrompt(
            system_prompt=DISTORTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def validate_distortion(
        self,
        original_text: str,
        distorted_text: str,
        task: Task,
    ) -> BenchmarkValidationResult:
        if not Mistral:
            return BenchmarkValidationResult(is_valid=True, reason="mistralai not installed — skipped")

        api_key = os.getenv(self.config.get("validation_api_key_env", "MISTRAL_API_KEY"))
        if not api_key:
            return BenchmarkValidationResult(is_valid=True, reason="MISTRAL_API_KEY not set — skipped")

        client = Mistral(api_key=api_key)
        model = self.config.get("validation_model", "mistral-large-latest")
        prompt = get_validation_prompt(original_text, distorted_text)

        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0,
            )
            content = response.choices[0].message.content.strip().lower()
            clean_content = content.replace('*', '').strip()
            is_yes = clean_content.startswith("yes")
            return BenchmarkValidationResult(is_valid=is_yes, reason=content)
        except Exception as e:
            return BenchmarkValidationResult(is_valid=False, reason=str(e))

    def get_generation_prompt(self, task: Task) -> Tuple[str, str]:
        question = (
            task.distorted_text
            if task.distorted_text is not None
            else task.data.get("question", "")
        )
        image_path = task.data.get("image_path", "")
        user_prompt = f"Image path: {image_path}\nQuestion: {question}\nAnswer:"
        return VQA_SYSTEM_PROMPT, user_prompt

    def evaluate(self, task: Task, response: str) -> EvalResult:
        ground_truth = task.data.get("answers", [])
        score = vqa_score(response, ground_truth)
        return EvalResult(
            task_id=task.task_id,
            is_correct=score > 0.0,
            metadata={
                "vqa_score": score,
                "predicted": response,
                "ground_truth": ground_truth,
                "miu": task.miu,
            },
        )

    def calculate_metrics(self, results: List[EvalResult]) -> BenchmarkMetrics:
        if not results:
            return BenchmarkMetrics(overall_score=0.0, per_mu={}, metadata={})

        scores = [r.metadata.get("vqa_score", 0.0) for r in results]
        overall = sum(scores) / len(scores)

        mu_buckets: Dict[float, List[float]] = {}
        for r in results:
            mu = r.metadata.get("miu")
            if mu is not None:
                mu_buckets.setdefault(float(mu), []).append(r.metadata.get("vqa_score", 0.0))
        per_mu = {mu: sum(v) / len(v) for mu, v in mu_buckets.items()}

        exact_match = sum(r.is_correct for r in results) / len(results)

        return BenchmarkMetrics(
            overall_score=overall,
            per_mu=per_mu,
            metadata={"vqa_accuracy": overall, "exact_match": exact_match, "n": len(results)},
        )

    # =========================================================================
    # Image helpers
    # =========================================================================

    def _encode_image_b64(self, image_path: str) -> Optional[str]:
        path = Path(image_path)
        if not path.exists():
            self.logger.warning(f"Image not found: {image_path}")
            return None
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # =========================================================================
    # Provider-specific generation calls
    # =========================================================================

    def _call_mistral_generate(
        self,
        client,
        model: str,
        system_prompt: str,
        question: str,
        image_path: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        model_info = ALLOWED_MODELS.get(model, {})
        if model_info.get("vision") and image_path:
            b64 = self._encode_image_b64(image_path)
            if b64:
                mime = _get_mime_type(image_path)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
                            {"type": "text", "text": f"Question: {question}\nAnswer:"},
                        ],
                    },
                ]
            else:
                messages = build_generation_messages(question, image_path)
        else:
            messages = build_generation_messages(question, image_path)

        response = self._call_with_retry(
            client.chat.complete,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_gemini_generate(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        question: str,
        image_path: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not _google_genai:
            raise ImportError("google-genai not installed. Run: pip install google-genai")

        client = _google_genai.Client(api_key=api_key)

        contents: List[Any] = []
        if image_path:
            path = Path(image_path)
            if path.exists():
                with path.open("rb") as f:
                    image_bytes = f.read()
                mime = _get_mime_type(image_path)
                contents.append(_google_types.Part.from_bytes(data=image_bytes, mime_type=mime))
            else:
                self.logger.warning(f"Image not found: {image_path}")

        contents.append(f"Question: {question}\nAnswer:")

        gen_config = _google_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self._call_with_retry(
            client.models.generate_content,
            model=model,
            contents=contents,
            config=gen_config,
        )
        if response.text is None:
            raise ValueError("Empty response from Gemini (likely safety filter or content block)")
        return response.text.strip()

    # =========================================================================
    # Pipeline helpers (internal, dict-based)
    # =========================================================================

    def _load_raw_tasks(self, data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        entries = load_entries(data_path, limit=limit)
        images_dir = Path(data_path).parent / "sampled_images_300"

        tasks: List[Dict[str, Any]] = []
        for item in entries:
            image_field = item.get("image")
            if image_field:
                filename = Path(image_field).name
            else:
                image_id = item.get("image_id", 0)
                filename = f"COCO_val2014_{int(image_id):012d}.jpg"
            image_path = str(images_dir / filename)

            raw_answers = item.get("answers", [])
            tasks.append(
                {
                    "question_id": str(item["question_id"]),
                    "question": item["question"],
                    "image_id": item.get("image_id"),
                    "image_path": image_path,
                    "answers": normalize_answers(raw_answers),
                    "multiple_choice_answer": item.get("multiple_choice_answer"),
                    "answer_type": item.get("answer_type"),
                }
            )

        self.logger.info(f"✓ Loaded {len(tasks)} tasks from {data_path}")
        return tasks

    def _distort_questions(
        self, tasks: List[Dict[str, Any]], miu: float
    ) -> List[Dict[str, Any]]:
        distorter = DistortionEngine.create(
            {
                "engine_type": self.config.get("distortion_engine", "api"),
                "vendor": self.config.get("distortion_vendor", "mistral"),
                "model_name": self.config.get("distortion_model", "mistral-large-latest"),
                "api_key_env_var": self.config.get("distortion_api_key_env", "MISTRAL_API_KEY"),
            },
            project_path=self.config.get("project_path"),
        )

        self.logger.info(f"🔀 Distorting {len(tasks)} questions with miu={miu}...")
        success_count = 0
        inter_request_delay = self.config.get("inter_request_delay", 3)

        for task in tasks:
            try:
                res = self._call_with_retry(
                    distorter.distort_question,
                    question_id=task["question_id"],
                    question=task["question"],
                    miu=miu,
                )
                task["distorted_question"] = res.distorted_question
                task["distortion_success"] = res.success
                task["distortion_miu"] = miu
                task["distortion_error"] = res.error
                if res.success:
                    success_count += 1
            except Exception as e:
                task["distorted_question"] = None
                task["distortion_success"] = False
                task["distortion_error"] = str(e)
                self.logger.error(f"  ⚠ {task['question_id']} - distortion error: {e}")
            time.sleep(inter_request_delay)

        self.logger.info(f"✅ Distorted {success_count}/{len(tasks)} tasks")
        return tasks

    def _validate_distortions(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not Mistral:
            self.logger.warning("mistralai not installed — skipping validation")
            for task in tasks:
                task["validation_passed"] = None
            return tasks

        api_key = os.getenv(self.config.get("validation_api_key_env", "MISTRAL_API_KEY"))
        if not api_key:
            self.logger.warning("MISTRAL_API_KEY not found — skipping validation")
            for task in tasks:
                task["validation_passed"] = None
            return tasks

        client = Mistral(api_key=api_key)
        model = self.config.get("validation_model", "mistral-large-latest")

        self.logger.info(f"🔍 Validating {len(tasks)} distortions...")
        passed = 0
        inter_request_delay = self.config.get("inter_request_delay", 3)

        for task in tasks:
            if not task.get("distortion_success"):
                task["validation_passed"] = False
                task["validation_reason"] = "distortion_failed"
                continue

            prompt = get_validation_prompt(task["question"], task.get("distorted_question", ""))
            try:
                response = self._call_with_retry(
                    client.chat.complete,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32,
                    temperature=0,
                )
                content = response.choices[0].message.content.strip().lower()
                clean_content = content.replace('*', '').strip()
                is_yes = clean_content.startswith("yes")
                task["validation_passed"] = is_yes
                task["validation_reason"] = content
                if is_yes:
                    passed += 1
            except Exception as e:
                task["validation_passed"] = None
                task["validation_reason"] = str(e)
                self.logger.error(f"  ⚠ {task['question_id']} — validation error: {e}")
            time.sleep(inter_request_delay)

        self.logger.info(f"✅ Validation passed: {passed}/{len(tasks)}")
        return tasks

    def _generate_answers(
        self, tasks: List[Dict[str, Any]], for_distorted: bool = False
    ) -> List[Dict[str, Any]]:
        model = self.config.get("generation_model", "gemini-2.5-flash")

        if model not in ALLOWED_MODELS:
            allowed = list(ALLOWED_MODELS.keys())
            self.logger.error(f"Model '{model}' not in ALLOWED_MODELS. Choose from: {allowed}")
            return tasks

        model_info = ALLOWED_MODELS[model]
        provider = model_info["provider"]
        api_key = os.getenv(model_info["api_key_env"])

        if not api_key:
            self.logger.warning(
                f"{model_info['api_key_env']} not set — skipping generation. "
                f"Add it to your .env file."
            )
            return tasks

        mistral_client = None
        if provider == "mistral":
            if not Mistral:
                self.logger.warning("mistralai not installed — skipping generation")
                return tasks
            mistral_client = Mistral(api_key=api_key)

        temperature = self.config.get("generation_temperature", 0.0)
        max_tokens = self.config.get("generation_max_tokens", 64)
        tag = "distorted" if for_distorted else "original"
        answer_field = f"{tag}_answer"
        success_field = f"{tag}_generation_success"

        self.logger.info(
            f"💻 Generating {tag} answers for {len(tasks)} tasks "
            f"[model={model}, provider={provider}]..."
        )

        success_count = 0
        delay = max(self.config.get("inter_request_delay", 4), 5)
        consecutive_successes = 0

        for task in tasks:
            if for_distorted:
                if not task.get("distortion_success") or not task.get("validation_passed"):
                    task[answer_field] = None
                    task[success_field] = False
                    continue
                question_text = task.get("distorted_question")
            else:
                question_text = task.get("question")

            if not question_text:
                task[answer_field] = None
                task[success_field] = False
                continue

            image_path = task.get("image_path", "")

            try:
                if provider == "mistral":
                    answer = self._call_mistral_generate(
                        mistral_client, model, VQA_SYSTEM_PROMPT,
                        question_text, image_path, temperature, max_tokens,
                    )
                else:
                    answer = self._call_gemini_generate(
                        api_key, model, VQA_SYSTEM_PROMPT,
                        question_text, image_path, temperature, max_tokens,
                    )
                task[answer_field] = answer
                task[success_field] = True
                success_count += 1
                consecutive_successes += 1
                drop = 10 if consecutive_successes == 1 else 5
                delay = max(delay - drop, 5)
            except Exception as e:
                task[answer_field] = None
                task[success_field] = False
                self.logger.error(f"  ⚠ {task['question_id']} — generation error: {e}")
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    consecutive_successes = 0
                    delay += 5
                    self.logger.warning(f"Rate limit hit — inter-request delay → {delay}s")

            time.sleep(delay)

        self.logger.info(f"✅ Generated {tag} answers: {success_count}/{len(tasks)}")
        return tasks

    def _evaluate_tasks(
        self, tasks: List[Dict[str, Any]]
    ) -> Tuple[List[Dict], List[Dict]]:
        original_results, distorted_results = [], []
        for task in tasks:
            gt = task.get("answers", [])
            qid = task["question_id"]
            if task.get("original_answer"):
                res = evaluate_answer(qid, task["original_answer"], gt)
                task["original_vqa_score"] = res["vqa_score"]
                original_results.append(res)
            if task.get("distorted_answer"):
                res = evaluate_answer(qid, task["distorted_answer"], gt)
                task["distorted_vqa_score"] = res["vqa_score"]
                distorted_results.append(res)
        return original_results, distorted_results

    # =========================================================================
    # Full pipeline
    # =========================================================================

    def run_full_pipeline(
        self,
        data_path: str,
        output_dir: str,
        miu: float = 0.6,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / "checkpoint.jsonl"

        tasks = []
        if checkpoint_path.exists():
            self.logger.info(f"🔄 Found checkpoint file: {checkpoint_path}")
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        tasks.append(json.loads(line))
            self.logger.info(f"✅ Resumed {len(tasks)} tasks from checkpoint.")
            
        else:
            self.logger.info("\n📚 [STEP 1/5] Loading data...")
            tasks = self._load_raw_tasks(data_path, limit=limit)

        def save_checkpoint():
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for task in tasks:
                    f.write(json.dumps(task, ensure_ascii=False) + "\n")

        # Check if we need to run Step 2 (Distortion)
        if any("distorted_question" not in t for t in tasks):
            self.logger.info("\n🔀 [STEP 2/5] Distorting questions...")
            tasks = self._distort_questions(tasks, miu=miu)
            save_checkpoint()
        else:
            self.logger.info("\n🔀 [STEP 2/5] Distortions already completed. Skipping.")

        # Check if we need to run Step 3 (Validation)
        if any("validation_passed" not in t for t in tasks):
            self.logger.info("\n🔍 [STEP 3/5] Validating distortions...")
            tasks = self._validate_distortions(tasks)
            save_checkpoint()
        else:
            self.logger.info("\n🔍 [STEP 3/5] Validations already completed. Skipping.")

        # Check if we need to run Step 4a (Original generation)
        if any(not t.get("original_answer") for t in tasks):
            self.logger.info("\n💻 [STEP 4a/5] Generating ORIGINAL answers...")
            tasks = self._generate_answers(tasks, for_distorted=False)
            save_checkpoint()
        else:
            self.logger.info("\n💻 [STEP 4a/5] Original answers already generated. Skipping.")

        # Check if we need to run Step 4b (Distorted generation)
        if any(not t.get("distorted_answer") for t in tasks):
            self.logger.info("\n💻 [STEP 4b/5] Generating DISTORTED answers...")
            tasks = self._generate_answers(tasks, for_distorted=True)
            save_checkpoint()
        else:
            self.logger.info("\n💻 [STEP 4b/5] Distorted answers already generated. Skipping.")

        self.logger.info("\n🏆 [STEP 5/5] Evaluating answers...")
        original_results, distorted_results = self._evaluate_tasks(tasks)

        original_metrics = compute_vqa_metrics(original_results)
        distorted_metrics = compute_vqa_metrics(distorted_results)

        summary = {
            "total_tasks": len(tasks),
            "distortion_successful": sum(1 for t in tasks if t.get("distortion_success")),
            "validation_passed": sum(1 for t in tasks if t.get("validation_passed") is True),
            "original_answers_generated": sum(1 for t in tasks if t.get("original_answer")),
            "distorted_answers_generated": sum(1 for t in tasks if t.get("distorted_answer")),
            "original_metrics": original_metrics,
            "distorted_metrics": distorted_metrics,
            "miu": miu,
        }

        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        tasks_filename = self.config.get("tasks_filename", "tasks_complete.jsonl")
        with open(output_path / tasks_filename, "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")

        self.logger.info("\n" + "=" * 70)
        self.logger.info("📊 Final Results:")
        self.logger.info(f"  Total tasks:          {summary['total_tasks']}")
        self.logger.info(f"  Distorted:            {summary['distortion_successful']}")
        self.logger.info(f"  Validated:            {summary['validation_passed']}")
        self.logger.info(f"  Original VQA score:   {original_metrics.get('vqa_accuracy', 0):.4f}")
        self.logger.info(f"  Distorted VQA score:  {distorted_metrics.get('vqa_accuracy', 0):.4f}")
        self.logger.info("=" * 70)

        return summary


# =============================================================================
# Convenience function
# =============================================================================

def run_vqav2_pipeline(
    data_path: str,
    output_dir: str,
    miu: float = 0.6,
    limit: Optional[int] = None,
    distortion_model: str = "mistral-large-latest",
    distortion_vendor: str = "mistral",
    generation_model: str = "gemini-2.5-flash",
    validation_model: str = "mistral-large-latest",
    validation_vendor: str = "mistral",
    project_path: Optional[str] = None,
    inter_request_delay: int = 3,
    tasks_filename: str = "tasks_complete.jsonl",
) -> Dict[str, Any]:
    config = {
        "distortion_model": distortion_model,
        "distortion_vendor": distortion_vendor,
        "distortion_api_key_env": f"{distortion_vendor.upper()}_API_KEY",
        "generation_model": generation_model,
        "validation_model": validation_model,
        "validation_vendor": validation_vendor,
        "validation_api_key_env": f"{validation_vendor.upper()}_API_KEY",
        "project_path": project_path,
        "inter_request_delay": inter_request_delay,
        "tasks_filename": tasks_filename,
    }
    benchmark = VQAV2Benchmark(config)
    return benchmark.run_full_pipeline(data_path, output_dir, miu=miu, limit=limit)
