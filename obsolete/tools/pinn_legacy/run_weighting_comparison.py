#!/usr/bin/env python3
"""Thin wrapper for the pipeline-style weighting comparison launcher."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiments.pipeline.run_weighting_comparison import main


if __name__ == "__main__":
    main()
