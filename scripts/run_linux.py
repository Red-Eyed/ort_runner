#!/usr/bin/env python3
"""Runs the native Linux build against a model."""
from __future__ import annotations

import argparse

from targets import Target, run_target_binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    run_target_binary(
        Target.LINUX, ["build-linux/bin/ort_runner", "--model", args.model, *args.args]
    )


if __name__ == "__main__":
    main()
