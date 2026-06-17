"""
Full VQAv2 robustness pipeline — 150-question run with Gemini 2.0 Flash.

Usage (from repo root):
    python benchmarks/vqav2/run_150_gemini.py

Results land in:
    benchmarks/vqav2/results/run_150_gemini/
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ── repo-root on sys.path ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vqav2.vqav2 import run_vqav2_pipeline

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_PATH   = SCRIPT_DIR / "data" / "vqav2_sampled_optimized.json"
OUTPUT_DIR  = SCRIPT_DIR / "results" / "run_150_gemini"

# ── config ────────────────────────────────────────────────────────────────────
N_QUESTIONS       = 150
MIU               = 0.5          # moderate distortion intensity
GENERATION_MODEL  = "gemini-2.5-flash"
DISTORTION_MODEL  = "mistral-large-latest"
VALIDATION_MODEL  = "mistral-large-latest"

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("VQAv2 Robustness Benchmark  —  150-question run")
    log.info(f"  Data      : {DATA_PATH}")
    log.info(f"  Output    : {OUTPUT_DIR}")
    log.info(f"  Questions : {N_QUESTIONS}")
    log.info(f"  mu        : {MIU}")
    log.info(f"  Distortion: {DISTORTION_MODEL} (Mistral)")
    log.info(f"  Validation: {VALIDATION_MODEL} (Mistral)")
    log.info(f"  Generation: {GENERATION_MODEL} (Gemini)")
    log.info("=" * 70)

    summary = run_vqav2_pipeline(
        data_path           = str(DATA_PATH),
        output_dir          = str(OUTPUT_DIR),
        miu                 = MIU,
        limit               = N_QUESTIONS,
        distortion_model    = DISTORTION_MODEL,
        distortion_vendor   = "mistral",
        generation_model    = GENERATION_MODEL,
        validation_model    = VALIDATION_MODEL,
        validation_vendor   = "mistral",
        inter_request_delay = 4,
        tasks_filename      = "tasks_complete_gemini.jsonl",
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for key, val in summary.items():
        print(f"  {key}: {val}")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
