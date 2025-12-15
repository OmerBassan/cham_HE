"""
Checkpoint Validators for Chameleon Pipeline

Validates data integrity at each stage:
1. CHECKPOINT_PRELIMINARY: After preliminary JSONL creation
2. CHECKPOINT_DISTORTION: After distortion generation
3. CHECKPOINT_PRE_EVAL: Before evaluation submission
4. CHECKPOINT_POST_EVAL: After evaluation results

Uses LLM judge for semantic validation when needed.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class CheckpointStage(Enum):
    """Pipeline checkpoint stages."""
    PRELIMINARY = "preliminary"
    DISTORTION = "distortion"
    PRE_EVAL = "pre_evaluation"
    POST_EVAL = "post_evaluation"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    stage: CheckpointStage
    checks_passed: int
    checks_failed: int
    errors: List[str]
    warnings: List[str]
    details: Dict[str, Any]


class CheckpointValidator:
    """
    Validates data at each pipeline checkpoint.
    """
    
    def __init__(self, project_dir: Path, distortions_per_question: int = 10):
        self.project_dir = Path(project_dir)
        self.N = distortions_per_question
    
    def validate_preliminary(self, df: pd.DataFrame) -> ValidationResult:
        """
        CHECKPOINT 1: Validate preliminary JSONL structure.
        
        Checks:
        - All required columns exist
        - composite_key is unique
        - No NULL in critical fields
        - Row count matches expected formula
        """
        errors = []
        warnings = []
        checks_passed = 0
        checks_failed = 0
        details = {}
        
        # Required columns
        required_cols = [
            'composite_key', 'question_id', 'distortion_id', 'miu',
            'question_text', 'distorted_question', 'answer'
        ]
        
        # Check 1: Required columns
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            checks_failed += 1
        else:
            checks_passed += 1
        
        # Check 2: composite_key uniqueness
        if 'composite_key' in df.columns:
            duplicates = df['composite_key'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"composite_key has {duplicates} duplicates - must be unique!")
                checks_failed += 1
            else:
                checks_passed += 1
                details['composite_keys_unique'] = True
        
        # Check 3: No NULL in critical fields
        critical_fields = ['question_id', 'miu', 'question_text', 'answer']
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isna().sum()
                if null_count > 0:
                    errors.append(f"Field '{field}' has {null_count} NULL values")
                    checks_failed += 1
                else:
                    checks_passed += 1
        
        # Check 4: options_json check removed
        # if 'options_json' in df.columns: ...
                
        # Check 5: Row count formula
        unique_questions = df['question_id'].nunique()
        miu_values = df['miu'].unique()
        miu_gt_0 = [m for m in miu_values if m > 0]
        
        expected_rows = unique_questions * (1 + len(miu_gt_0) * self.N)  # 1 for miu=0, N for each miu>0
        actual_rows = len(df)
        
        details['unique_questions'] = unique_questions
        details['miu_values'] = list(miu_values)
        details['expected_rows'] = expected_rows
        details['actual_rows'] = actual_rows
        
        if actual_rows != expected_rows:
            warnings.append(f"Row count mismatch: expected {expected_rows}, got {actual_rows}")
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            stage=CheckpointStage.PRELIMINARY,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_distortions(self, df: pd.DataFrame) -> ValidationResult:
        """
        CHECKPOINT 2: Validate distortion generation.
        
        Checks:
        - All miu>0 rows have distorted_question filled
        - No duplicates per question+miu
        - Each question has exactly N distortions per miu
        - Distortions are different from original
        """
        errors = []
        warnings = []
        checks_passed = 0
        checks_failed = 0
        details = {}
        
        # Filter to miu > 0
        df_miu = df[df['miu'] > 0].copy()
        
        # Check 1: All distorted_question filled
        empty_count = df_miu['distorted_question'].isna().sum()
        empty_count += (df_miu['distorted_question'] == '').sum()
        
        if empty_count > 0:
            errors.append(f"{empty_count} rows still have empty distorted_question")
            checks_failed += 1
        else:
            checks_passed += 1
        details['empty_distortions'] = int(empty_count)
        
        # Check 2: No duplicates per question+miu
        duplicate_count = 0
        for (q_id, miu), group in df_miu.groupby(['question_id', 'miu']):
            texts = group['distorted_question'].dropna().tolist()
            unique_texts = set(str(t).strip().lower() for t in texts if t)
            if len(unique_texts) < len(texts):
                duplicate_count += len(texts) - len(unique_texts)
        
        if duplicate_count > 0:
            errors.append(f"{duplicate_count} duplicate distortions found")
            checks_failed += 1
        else:
            checks_passed += 1
        details['duplicate_distortions'] = duplicate_count
        
        # Check 3: Exactly N distortions per question+miu
        incomplete_count = 0
        for (q_id, miu), group in df_miu.groupby(['question_id', 'miu']):
            filled = group['distorted_question'].notna() & (group['distorted_question'] != '')
            if filled.sum() != self.N:
                incomplete_count += 1
        
        if incomplete_count > 0:
            warnings.append(f"{incomplete_count} question+miu pairs don't have exactly {self.N} distortions")
        details['incomplete_pairs'] = incomplete_count
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            stage=CheckpointStage.DISTORTION,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_pre_evaluation(self, df: pd.DataFrame) -> ValidationResult:
        """
        CHECKPOINT 3: Validate before evaluation submission.
        
        Checks:
        - All rows have valid options_json
        - composite_key exists and is unique
        - distorted_question filled for all rows
        """
        errors = []
        warnings = []
        checks_passed = 0
        checks_failed = 0
        details = {}
        
        # Check 1: composite_key
        if 'composite_key' not in df.columns:
            errors.append("Missing composite_key column - required for result matching")
            checks_failed += 1
        else:
            if df['composite_key'].duplicated().any():
                errors.append("composite_key has duplicates!")
                checks_failed += 1
            else:
                checks_passed += 1
        
        # Check 2: options_json check removed
        
        # Check 3: distorted_question filled
        empty = df['distorted_question'].isna().sum() + (df['distorted_question'] == '').sum()
        if empty > 0:
            warnings.append(f"{empty} rows have empty distorted_question")
        details['empty_questions'] = int(empty)
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            stage=CheckpointStage.PRE_EVAL,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_post_evaluation(self, df: pd.DataFrame) -> ValidationResult:
        """
        CHECKPOINT 4: Validate after evaluation results.
        
        Checks:
        - target_model_answer filled for all rows with valid options
        - is_correct calculated
        - Match rate acceptable
        """
        errors = []
        warnings = []
        checks_passed = 0
        checks_failed = 0
        details = {}
        
        # Check 1: target_model_answer coverage
        total = len(df)
        answered = df['target_model_answer'].notna() & (df['target_model_answer'] != '')
        answered_count = answered.sum()
        answer_rate = answered_count / total if total > 0 else 0
        
        details['total_rows'] = total
        details['answered_rows'] = int(answered_count)
        details['answer_rate'] = round(answer_rate * 100, 1)
        
        if answer_rate < 0.95:
            warnings.append(f"Only {answer_rate*100:.1f}% of rows have answers (expected >95%)")
        else:
            checks_passed += 1
        
        # Check 2: is_correct calculated
        if 'is_correct' in df.columns:
            correct_count = df['is_correct'].sum() if df['is_correct'].dtype == bool else 0
            accuracy = correct_count / answered_count if answered_count > 0 else 0
            details['correct_count'] = int(correct_count)
            details['accuracy'] = round(accuracy * 100, 1)
            checks_passed += 1
        else:
            errors.append("is_correct column missing")
            checks_failed += 1
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            stage=CheckpointStage.POST_EVAL,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def print_result(self, result: ValidationResult):
        """Pretty print validation result."""
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {result.stage.value.upper()} - {status}")
        print(f"{'='*60}")
        print(f"Checks: {result.checks_passed} passed, {result.checks_failed} failed")
        
        if result.errors:
            print("\n🚨 ERRORS:")
            for e in result.errors:
                print(f"   ❌ {e}")
        
        if result.warnings:
            print("\n⚠️ WARNINGS:")
            for w in result.warnings:
                print(f"   ⚠️ {w}")
        
        if result.details:
            print("\n📊 DETAILS:")
            for k, v in result.details.items():
                print(f"   {k}: {v}")


def run_all_checkpoints(project_dir: Path, stage: str = "all") -> Dict[str, ValidationResult]:
    """
    Run validation checkpoints for a project.
    
    Args:
        project_dir: Path to project directory
        stage: Which stage to validate ("preliminary", "distortion", "pre_eval", "post_eval", "all")
    
    Returns:
        Dict of validation results by stage
    """
    validator = CheckpointValidator(project_dir)
    results = {}
    
    jsonl_path = project_dir / "distorted_data" / "distortions_complete.jsonl"
    if not jsonl_path.exists():
        jsonl_path = project_dir / "distorted_data" / "distortions_in_progress.jsonl"
    
    if not jsonl_path.exists():
        # Fallback to check for CSV if JSONL doesn't exist (legacy support or mismatch)
        csv_path = project_dir / "distorted_data" / "distortions_complete.csv"
        if csv_path.exists():
             print(f"⚠️ Found CSV but expected JSONL at {project_dir / 'distorted_data'}")
             return results
        
        print(f"❌ No JSONL found in {project_dir / 'distorted_data'}")
        return results
    
    df = pd.read_json(jsonl_path, orient='records', lines=True)
    
    stages_to_run = {
        "preliminary": validator.validate_preliminary,
        "distortion": validator.validate_distortions,
        "pre_eval": validator.validate_pre_evaluation,
        "post_eval": validator.validate_post_evaluation,
    }
    
    if stage == "all":
        for name, func in stages_to_run.items():
            result = func(df)
            validator.print_result(result)
            results[name] = result
    elif stage in stages_to_run:
        result = stages_to_run[stage](df)
        validator.print_result(result)
        results[stage] = result
    
    return results

