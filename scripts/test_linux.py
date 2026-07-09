#!/usr/bin/env python3
"""Runs a Linux build's unit tests (defaults to the x86_64 build)."""

from __future__ import annotations

import argparse

from targets import Target, resolve, run_target_binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Target,
        choices=[Target.LINUX_X64, Target.LINUX_ARM64],
        nargs="?",
        default=Target.LINUX_X64,
    )
    args = parser.parse_args()

    config = resolve(args.target)
    run_target_binary(args.target, [f"{config.build_dir.name}/tests/ort_runner_tests"])


if __name__ == "__main__":
    main()
