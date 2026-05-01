"""Temporary compatibility wrapper for the moved experiment campaign entrypoint."""

from src.experiments.pipeline.run_campaign import *  # noqa: F401,F403
from src.experiments.pipeline.run_campaign import main


if __name__ == "__main__":
    main()
