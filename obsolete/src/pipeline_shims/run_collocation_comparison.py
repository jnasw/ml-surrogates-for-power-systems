"""Temporary compatibility wrapper for the moved collocation comparison launcher."""

from src.experiments.pipeline.run_collocation_comparison import *  # noqa: F401,F403
from src.experiments.pipeline.run_collocation_comparison import main


if __name__ == "__main__":
    main()
