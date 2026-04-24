"""Temporary compatibility wrapper for the moved optimizer comparison launcher."""

from src.experiments.pipeline.run_optimizer_comparison import *  # noqa: F401,F403
from src.experiments.pipeline.run_optimizer_comparison import main


if __name__ == "__main__":
    main()
