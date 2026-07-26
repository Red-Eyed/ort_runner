#!/usr/bin/env python3
"""Builds the containerized toolchain image for a target ('linux' or 'android')."""

from __future__ import annotations

import argparse
import subprocess

from targets import Target, container_engine, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()

    # Both Linux targets share one Containerfile and one multi-arch base image, so --platform
    # alone selects the architecture -- no per-arch base to keep in sync.
    config = resolve(args.target)
    subprocess.run(
        [
            container_engine(),
            "build",
            "--platform",
            config.image_platform,
            "-t",
            config.image_tag,
            "-f",
            str(config.containerfile),
            str(config.containerfile.parent),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
