"""
Quick smoke-test: run the VQAv2 pipeline on 3 samples.

Usage (from repo root):
    python benchmarks/vqav2/examples/test_consolidated.py
"""

from __future__ import annotations

from pathlib import Path
import sys

# Add repo root to sys.path to allow running this script directly from repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from benchmarks.vqav2.vqav2 import run_vqav2_pipeline

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "vqav2_sampled_optimized.json"
OUTPUT_DIR = SCRIPT_DIR / "results"


def main() -> None:
    results = run_vqav2_pipeline(
        data_path=str(DATA_PATH),
        output_dir=str(OUTPUT_DIR),
        miu=0.5,
        limit=3,
        generation_model="mistral-large-latest",
    )

    print("\n=== Test Run Summary ===")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
