#!/usr/bin/env python3
"""
Chameleon CLI - Main Entry Point

LLM Robustness Testing Framework
Evaluate model performance under various distortion conditions.

Usage:
    python cli.py init --name my_project --modality text
    python cli.py list
    python cli.py analyze --project my_project
    python cli.py help

For interactive project creation:
    python cli.py init
"""

import sys
from pathlib import Path

# Add the package to path for development
sys.path.insert(0, str(Path(__file__).parent))

from chameleon.cli.commands import main

if __name__ == "__main__":
    sys.exit(main())


