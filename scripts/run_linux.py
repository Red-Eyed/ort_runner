#!/usr/bin/env python3
"""Runs a Linux build against a model.

The arch (linux-x64 / linux-aarch64) is passed as the first argument so the aarch64 build can
be exercised too -- on a non-matching host it runs inside the target's (emulated) build image.
"""

from __future__ import annotations

import argparse

from targets import Target, resolve, run_target_binary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=[Target.LINUX_X64, Target.LINUX_ARM64])
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = resolve(args.target)
    runner = f"{config.build_dir.name}/bin/ort_runner"
    run_target_binary(args.target, [runner, "--model", args.model, *args.args])


if __name__ == "__main__":
    main()
