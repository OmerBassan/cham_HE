"""Tests for the VQAv2 benchmark."""

import json
import pytest
import tempfile
from pathlib import Path

from benchmarks.vqav2.vqav2 import VQAV2Benchmark
from benchmarks.vqav2.engine.execution import vqa_score, evaluate_answer
from benchmarks.vqav2.engine.evaluate_functional_correctness import compute_vqa_metrics
from benchmarks.vqav2.engine.data import load_entries, normalize_answers
from benchmarks.vqav2.prompts import get_distortion_prompt, get_validation_prompt, get_generation_prompt
from benchmarks.base.types import Task, EvalResult


SAMPLE_ENTRY = {
    "question_id": 1,
    "question": "What color is the sky?",
    "image_id": 123,
    "answers": ["blue", "blue", "blue", "light blue", "blue", "blue", "blue", "blue", "blue", "blue"],
    "multiple_choice_answer": "blue",
    "answer_type": "other",
}

SAMPLE_TASK_DATA = {
    "question_id": "1",
    "question": "What color is the sky?",
    "image_id": 123,
    "image_path": "/fake/path/image.jpg",
    "answers": ["blue", "blue", "blue", "light blue", "blue", "blue", "blue", "blue", "blue", "blue"],
    "multiple_choice_answer": "blue",
    "answer_type": "other",
}


@pytest.fixture
def benchmark():
    return VQAV2Benchmark(config={})


@pytest.fixture
def sample_task():
    return Task(task_id="1", data=SAMPLE_TASK_DATA)


@pytest.fixture
def sample_data_file(tmp_path):
    data_file = tmp_path / "vqav2_test.json"
    entries = [SAMPLE_ENTRY, {**SAMPLE_ENTRY, "question_id": 2, "question": "How many cats are there?"}]
    data_file.write_text(json.dumps(entries), encoding="utf-8")
    return str(data_file)


# =============================================================================
# engine/execution.py
# =============================================================================

class TestVqaScore:
    def test_exact_match_scores_full(self):
        gt = ["blue"] * 10
        assert vqa_score("blue", gt) == 1.0

    def test_partial_match(self):
        gt = ["blue", "blue", "blue", "sky blue", "light blue", "blue", "blue", "blue", "blue", "blue"]
        score = vqa_score("blue", gt)
        assert score == 1.0

    def test_one_match_out_of_ten(self):
        gt = ["blue"] + ["red"] * 9
        score = vqa_score("blue", gt)
        assert score == pytest.approx(1 / 3)

    def test_no_match(self):
        gt = ["red"] * 10
        assert vqa_score("blue", gt) == 0.0

    def test_case_insensitive(self):
        gt = ["Blue", "BLUE", "blue"]
        assert vqa_score("blue", gt) == 1.0

    def test_empty_ground_truth(self):
        assert vqa_score("blue", []) == 0.0

    def test_score_capped_at_one(self):
        gt = ["blue"] * 10
        assert vqa_score("blue", gt) <= 1.0


class TestEvaluateAnswer:
    def test_returns_dict_shape(self):
        result = evaluate_answer("1", "blue", ["blue"] * 10)
        assert "question_id" in result
        assert "vqa_score" in result
        assert "passed" in result
        assert "predicted" in result
        assert "ground_truth" in result

    def test_correct_answer_passed_true(self):
        result = evaluate_answer("1", "blue", ["blue"] * 10)
        assert result["passed"] is True

    def test_wrong_answer_passed_false(self):
        result = evaluate_answer("1", "green", ["blue"] * 10)
        assert result["passed"] is False


# =============================================================================
# engine/evaluate_functional_correctness.py
# =============================================================================

class TestComputeVqaMetrics:
    def test_empty_returns_zeros(self):
        metrics = compute_vqa_metrics([])
        assert metrics["vqa_accuracy"] == 0.0
        assert metrics["total"] == 0

    def test_all_correct(self):
        results = [{"vqa_score": 1.0, "passed": True}] * 5
        metrics = compute_vqa_metrics(results)
        assert metrics["vqa_accuracy"] == 1.0
        assert metrics["exact_match"] == 1.0
        assert metrics["total"] == 5

    def test_mixed(self):
        results = [
            {"vqa_score": 1.0, "passed": True},
            {"vqa_score": 0.0, "passed": False},
        ]
        metrics = compute_vqa_metrics(results)
        assert metrics["vqa_accuracy"] == pytest.approx(0.5)
        assert metrics["exact_match"] == pytest.approx(0.5)


# =============================================================================
# engine/data.py
# =============================================================================

class TestLoadEntries:
    def test_loads_list(self, sample_data_file):
        entries = load_entries(sample_data_file)
        assert len(entries) == 2

    def test_limit_respected(self, sample_data_file):
        entries = load_entries(sample_data_file, limit=1)
        assert len(entries) == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_entries("/nonexistent/path.json")

    def test_not_list_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON list"):
            load_entries(str(bad))

    def test_missing_required_field_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required fields"):
            load_entries(str(bad))


class TestNormalizeAnswers:
    def test_plain_strings(self):
        assert normalize_answers(["yes", "no"]) == ["yes", "no"]

    def test_dicts_with_answer_key(self):
        assert normalize_answers([{"answer": "yes"}, {"answer": "no"}]) == ["yes", "no"]

    def test_mixed(self):
        assert normalize_answers(["yes", {"answer": "no"}]) == ["yes", "no"]


# =============================================================================
# VQAV2Benchmark — AbstractBenchmark interface
# =============================================================================

class TestVQAV2Benchmark:
    def test_load_data(self, benchmark, sample_data_file):
        tasks = benchmark.load_data(sample_data_file)
        assert len(tasks) == 2
        assert all(hasattr(t, "task_id") for t in tasks)
        assert all(hasattr(t, "data") for t in tasks)

    def test_load_data_task_ids_are_strings(self, benchmark, sample_data_file):
        tasks = benchmark.load_data(sample_data_file)
        assert all(isinstance(t.task_id, str) for t in tasks)

    def test_get_field_to_distort(self, benchmark):
        assert benchmark.get_field_to_distort() == "question"

    def test_get_distortion_prompt_structure(self, benchmark, sample_task):
        dp = benchmark.get_distortion_prompt(sample_task, miu=0.6, n_distortions=1)
        assert dp.system_prompt
        assert dp.user_prompt
        assert "What color is the sky?" in dp.user_prompt

    def test_get_generation_prompt_original(self, benchmark, sample_task):
        system, user = benchmark.get_generation_prompt(sample_task)
        assert system
        assert "What color is the sky?" in user

    def test_get_generation_prompt_uses_distorted_text(self, benchmark, sample_task):
        sample_task.distorted_text = "What hue does the sky appear?"
        _, user = benchmark.get_generation_prompt(sample_task)
        assert "What hue does the sky appear?" in user

    def test_evaluate_correct(self, benchmark, sample_task):
        result = benchmark.evaluate(sample_task, "blue")
        assert isinstance(result, EvalResult)
        assert result.task_id == "1"
        assert result.is_correct is True
        assert "vqa_score" in result.metadata

    def test_evaluate_incorrect(self, benchmark, sample_task):
        result = benchmark.evaluate(sample_task, "green")
        assert result.is_correct is False

    def test_calculate_metrics_empty(self, benchmark):
        metrics = benchmark.calculate_metrics([])
        assert metrics.overall_score == 0.0
        assert isinstance(metrics.per_mu, dict)

    def test_calculate_metrics_shape(self, benchmark):
        results = [
            EvalResult(task_id="1", is_correct=True, metadata={"vqa_score": 1.0, "miu": 0.0}),
            EvalResult(task_id="2", is_correct=False, metadata={"vqa_score": 0.0, "miu": 0.6}),
        ]
        metrics = benchmark.calculate_metrics(results)
        assert 0.0 <= metrics.overall_score <= 1.0
        assert isinstance(metrics.per_mu, dict)
        assert 0.0 in metrics.per_mu
        assert 0.6 in metrics.per_mu

    def test_calculate_metrics_overall_score(self, benchmark):
        results = [
            EvalResult(task_id="1", is_correct=True, metadata={"vqa_score": 1.0}),
            EvalResult(task_id="2", is_correct=True, metadata={"vqa_score": 1.0}),
        ]
        metrics = benchmark.calculate_metrics(results)
        assert metrics.overall_score == pytest.approx(1.0)
