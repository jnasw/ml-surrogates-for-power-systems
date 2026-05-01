"""Temporary compatibility wrapper for the moved weighting comparison launcher."""

from src.experiments.pipeline.run_weighting_comparison import *  # noqa: F401,F403
from src.experiments.pipeline.run_weighting_comparison import main


if __name__ == "__main__":
    main()
