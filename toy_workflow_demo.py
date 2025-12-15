#!/usr/bin/env python3
"""
Toy Workflow Demo for Chameleon
Runs a comprehensive end-to-end test on just 3 samples to verify pipeline.
"""

import os
import sys
import shutil
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from chameleon.workflow import run_workflow

def setup_toy_project():
    """Create a minimal ToyRun project with 3 samples."""
    print(">> Setting up ToyRun project...")
    
    projects_dir = PROJECT_ROOT / "Projects"
    toy_dir = projects_dir / "ToyRun"
    
    # Clean/Create ToyRun
    if toy_dir.exists():
        shutil.rmtree(toy_dir)
    toy_dir.mkdir(parents=True, exist_ok=True)
    (toy_dir / "original_data").mkdir(exist_ok=True)

    # 1. Create Raw Data (Self-contained)
    print(f"   + Generating 3 raw HumanEval-style samples...")
    
    # 3 Real samples from HumanEval for authentic testing
    toy_data = [
        {
            "task_id": "HumanEval/0",
            "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0], 0.3)\n    True\n    \"\"\"\n",
            "entry_point": "has_close_elements",
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n"
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "from typing import List\n\n\ndef separate_paren_groups(paren_string: str) -> List[str]:\n    \"\"\" Input to this function is a string containing multiple groups of nested parentheses.\n    Your goal is to separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups('( ) (( )) (( )( ))')\n    ['()', '(())', '(()())']\n    \"\"\"\n",
            "entry_point": "separate_paren_groups",
            "canonical_solution": "    result = []\n    current_string = []\n    current_depth = 0\n    for c in paren_string:\n        if c == '(':\n            current_depth += 1\n            current_string.append(c)\n        elif c == ')':\n            current_depth -= 1\n            current_string.append(c)\n\n            if current_depth == 0:\n                result.append(''.join(current_string))\n                current_string.clear()\n\n    return result\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n    assert candidate('(()(())((())))') == ['(()(())((())))']\n    assert candidate('( ) (( )) (( )( ))') == ['()', '(())', '(()())']\n"
        },
        {
            "task_id": "HumanEval/2",
            "prompt": "def truncate_number(number: float) -> float:\n    \"\"\" Given a positive floating point number, it can be decomposed into\n    and integer part (largest integer smaller than given number) and decimals\n    (leftover part always smaller than 1).\n\n    Return the decimal part of the number.\n    >>> truncate_number(3.5)\n    0.5\n    \"\"\"\n",
            "entry_point": "truncate_number",
            "canonical_solution": "    return number % 1.0\n",
            "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n    assert abs(candidate(123.456) - 0.456) < 1e-6\n"
        }
    ]
            
    with open(toy_dir / "original_data" / "toy_data.jsonl", 'w', encoding='utf-8') as f:
        for item in toy_data:
            f.write(json.dumps(item) + "\n")
            
    # 2. Create Config (Minimal)
    print("   + Creating minimal config...")
    config = {
        "project_name": "ToyRun",
        "modality": "text",
        "target_model": {
            "vendor": "mistral",
            "name": "mistral-small-latest",
            "backend": "openai" # Mistral behaves like OpenAI in this framework
        },
        "distortion": {
            "miu_values": [0.5], # Single distortion level
            "distortions_per_question": 1, # Single variant
            "engine": {
                "vendor": "mistral",
                "model_name": "mistral-small-latest",
                "engine_type": "api"
            }
        }
    }
    
    with open(toy_dir / "config.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
        
    # 3. Copy .env (API Keys)
    # Try different locations for .env
    env_sources = [
        projects_dir / "Run2" / ".env",
        projects_dir / "TestRun1" / ".env",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "chameleon" / ".env"
    ]
    
    env_content = ""
    found_env = False
    
    # Also check os.environ for MISTRAL_API_KEY
    if "MISTRAL_API_KEY" in os.environ:
         env_content += f"MISTRAL_API_KEY={os.environ['MISTRAL_API_KEY']}\n"
         found_env = True
    
    for env_path in env_sources:
        if env_path.exists():
            print(f"   > Found .env at {env_path}")
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "MISTRAL_API_KEY" in content or "OPENAI_API_KEY" in content:
                        env_content += "\n" + content
                        found_env = True
                        break # Found a good one
            except:
                pass
                
    if not found_env and not env_content:
        print("   ! No API keys found! Distortion/Generation might fail.")
    
    with open(toy_dir / ".env", 'w', encoding='utf-8') as f:
        f.write(env_content)

    return True

def main():
    if not setup_toy_project():
        print("Failed to setup toy project.")
        return
        
    print("\n>> Starting Toy Workflow (ToyRun)...")
    print("=" * 60)
    
    try:
        run_workflow(
            project_name="ToyRun",
            projects_dir="Projects",
            skip_distortion=False,
            skip_generation=False,
            skip_evaluation=False,
            skip_analysis=False
        )
        print("\n[OK] Toy Workflow Completed Successfully!")
        print("   Check Projects/ToyRun/analysis/ for results.")
        
    except Exception as e:
        print(f"\n[ERROR] Toy Workflow Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
