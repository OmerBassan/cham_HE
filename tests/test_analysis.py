"""Tests for analysis module functionality."""

import pytest
import pandas as pd
import numpy as np

from chameleon.analysis.metrics import (
    calculate_accuracy,
    calculate_accuracy_by_group,
    calculate_degradation,
    clean_model_answer,
)
from chameleon.analysis.mcnemar import (
    mcnemar_test,
    McNemarResult,
    calculate_confidence_interval,
)


class TestMetrics:
    """Tests for metrics calculations."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame for testing."""
        return pd.DataFrame({
            "question_id": range(10),
            "model_answer": ["A", "B", "A", "C", "D", "A", "B", "C", "D", "A"],
            "correct_answer": ["A", "B", "A", "C", "A", "A", "B", "C", "D", "B"],
            "is_correct": [True, True, True, True, False, True, True, True, True, False],
            "miu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            "subject": ["math", "math", "physics", "physics", "math", 
                       "math", "math", "physics", "physics", "math"],
        })
    
    def test_calculate_accuracy_from_is_correct(self, sample_df):
        """Test accuracy calculation using is_correct column."""
        accuracy = calculate_accuracy(sample_df)
        
        assert accuracy == 0.8  # 8/10 correct
    
    def test_calculate_accuracy_from_comparison(self, sample_df):
        """Test accuracy calculation comparing answer columns."""
        # Remove is_correct to force comparison
        df = sample_df.drop(columns=["is_correct"])
        accuracy = calculate_accuracy(df, is_correct_col=None)
        
        assert accuracy == 0.8
    
    def test_accuracy_by_group(self, sample_df):
        """Test group-wise accuracy calculation."""
        result = calculate_accuracy_by_group(sample_df, "miu")
        
        assert len(result) == 2  # Two miu levels
        
        baseline_acc = result[result["miu"] == 0.0]["accuracy"].values[0]
        distorted_acc = result[result["miu"] == 0.5]["accuracy"].values[0]
        
        assert baseline_acc == 0.8  # 4/5 correct at miu=0.0
        assert distorted_acc == 0.8  # 4/5 correct at miu=0.5
    
    def test_calculate_degradation(self, sample_df):
        """Test degradation calculation."""
        result = calculate_degradation(
            sample_df,
            baseline_filter={"miu": 0.0},
            comparison_col="miu"
        )
        
        assert len(result) == 2
        
        # Baseline should have 0 degradation
        baseline_row = result[result["miu"] == 0.0].iloc[0]
        assert baseline_row["degradation"] == 0.0
    
    def test_clean_model_answer_simple(self):
        """Test cleaning simple answers."""
        assert clean_model_answer("A") == "A"
        assert clean_model_answer("B") == "B"
        assert clean_model_answer("  C  ") == "C"
    
    def test_clean_model_answer_extraction(self):
        """Test extracting answers from longer responses."""
        assert clean_model_answer("The answer is A") == "A"
        assert clean_model_answer("I think B is correct") == "B"
        assert clean_model_answer("Option C") == "C"
    
    def test_clean_model_answer_lowercase(self):
        """Test handling lowercase answers."""
        assert clean_model_answer("a") == "A"
        assert clean_model_answer("answer: b") == "B"
    
    def test_clean_model_answer_empty(self):
        """Test handling empty/nan values."""
        assert clean_model_answer("") == ""
        assert clean_model_answer(np.nan) == np.nan


class TestMcNemar:
    """Tests for McNemar's test implementation."""
    
    def test_mcnemar_significant_difference(self):
        """Test McNemar's test with significant difference."""
        # Group 1 better than Group 2
        group1 = np.array([True, True, True, True, True, False, True, True, True, True] * 10)
        group2 = np.array([True, True, True, False, False, False, True, True, False, True] * 10)
        
        result = mcnemar_test(group1, group2)
        
        assert isinstance(result, McNemarResult)
        assert result.group1_accuracy > result.group2_accuracy
        assert result.accuracy_difference > 0
    
    def test_mcnemar_no_difference(self):
        """Test McNemar's test with no difference."""
        # Same results for both groups
        group1 = np.array([True, True, False, False, True, True, False, True, True, True])
        group2 = group1.copy()
        
        result = mcnemar_test(group1, group2)
        
        assert result.accuracy_difference == 0
        assert result.discordant_pairs == 0
    
    def test_mcnemar_result_dict(self):
        """Test McNemarResult to_dict method."""
        group1 = np.array([True, True, False, True, True])
        group2 = np.array([True, False, False, True, False])
        
        result = mcnemar_test(group1, group2)
        result_dict = result.to_dict()
        
        assert "statistic" in result_dict
        assert "p_value" in result_dict
        assert "significance" in result_dict
    
    def test_confidence_interval(self):
        """Test Wilson score confidence interval."""
        # 80% accuracy with 100 samples
        lower, upper = calculate_confidence_interval(0.8, 100)
        
        assert lower < 0.8
        assert upper > 0.8
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1
    
    def test_confidence_interval_edge_cases(self):
        """Test confidence interval edge cases."""
        # 0 samples
        lower, upper = calculate_confidence_interval(0.5, 0)
        assert lower == 0
        assert upper == 0
        
        # 100% accuracy
        lower, upper = calculate_confidence_interval(1.0, 100)
        assert upper == 1.0
        
        # 0% accuracy
        lower, upper = calculate_confidence_interval(0.0, 100)
        assert lower == 0.0


class TestAnalysisIntegration:
    """Integration tests for analysis pipeline."""
    
    @pytest.fixture
    def full_dataset(self):
        """Create a more realistic dataset for integration tests."""
        np.random.seed(42)
        
        subjects = ["math", "physics", "biology", "history"]
        mius = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        data = []
        for subject in subjects:
            base_accuracy = np.random.uniform(0.85, 0.95)
            
            for miu in mius:
                # Simulate degradation with miu
                degradation = miu * np.random.uniform(0.1, 0.3)
                accuracy = base_accuracy - degradation
                
                for i in range(50):
                    is_correct = np.random.random() < accuracy
                    data.append({
                        "question_id": f"q_{subject}_{miu}_{i}",
                        "subject": subject,
                        "miu": miu,
                        "is_correct": is_correct,
                        "model_answer": "A" if is_correct else "B",
                        "correct_answer": "A",
                    })
        
        return pd.DataFrame(data)
    
    def test_full_analysis_pipeline(self, full_dataset):
        """Test running full analysis pipeline."""
        from chameleon.analysis.mcnemar import (
            analyze_distortion_significance,
            analyze_subject_significance,
        )
        
        # Distortion analysis
        distortion_results = analyze_distortion_significance(
            full_dataset,
            baseline_col="miu",
            baseline_value=0.0,
            is_correct_col="is_correct"
        )
        
        assert len(distortion_results) == 9  # 9 non-baseline levels
        assert "p_value" in distortion_results.columns
        assert "significance" in distortion_results.columns
        
        # Subject analysis
        subject_results = analyze_subject_significance(
            full_dataset,
            subject_col="subject",
            baseline_col="miu",
            baseline_value=0.0,
            comparison_value=0.9,
            is_correct_col="is_correct"
        )
        
        assert len(subject_results) == 4  # 4 subjects
        assert "degradation_percent" in subject_results.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


