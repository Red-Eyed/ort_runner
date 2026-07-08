#!/usr/bin/env python3
"""Runs the native Linux build locally against a model."""
from __future__ import annotations

import argparse
import os
import subprocess

from targets import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    build_dir = REPO_ROOT / "build-linux"
    env = os.environ | {"LD_LIBRARY_PATH": str(build_dir / "bin")}
    subprocess.run(
        [str(build_dir / "bin" / "ort_runner"), "--model", args.model, *args.args],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
