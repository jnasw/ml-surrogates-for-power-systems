"""Temporary compatibility wrapper for the moved experiment pipeline entrypoint."""

from src.experiments.pipeline.run_experiment import *  # noqa: F401,F403
from src.experiments.pipeline.run_experiment import main


if __name__ == "__main__":
    main()
