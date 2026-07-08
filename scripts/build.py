#!/usr/bin/env python3
"""Fetches ONNX Runtime, configures, and builds a target inside its podman toolchain image.

Assumes the image has already been built (see build_image.py).
"""
from __future__ import annotations

import argparse
import subprocess

from targets import REPO_ROOT, Target, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()

    config = resolve(args.target)
    platform_args = ["--platform", config.run_platform] if config.run_platform else []
    build_dir_name = config.build_dir.name
    subprocess.run(
        [
            "podman", "run", "--rm",
            *platform_args,
            "-v", f"{REPO_ROOT}:/workspace:Z",
            "-w", "/workspace",
            config.image_tag,
            "bash", "-c",
            f"scripts/fetch_onnxruntime.sh {config.fetch_arg} "
            f"&& cmake --preset {config.cmake_preset} -B {build_dir_name} "
            f"&& cmake --build {build_dir_name} -j",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
