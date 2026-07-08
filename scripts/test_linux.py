#!/usr/bin/env python3
"""Runs the native Linux build's unit tests."""
from __future__ import annotations

from targets import Target, run_target_binary


def main() -> None:
    run_target_binary(Target.LINUX, ["build-linux/tests/ort_runner_tests"])


if __name__ == "__main__":
    main()
