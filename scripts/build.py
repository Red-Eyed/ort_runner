#!/usr/bin/env python3
"""Fetches ONNX Runtime, then builds a target inside its podman toolchain image.

The two halves run in different places on purpose:

  * fetching runs on the host, because it only downloads into the bind-mounted repo. Keeping it
    out of the container means the images need no Python at all -- and uv's x86_64 binary
    segfaults under the QEMU emulation the Android image is always built with.
  * compiling runs in the container, so no Rust toolchain, NDK or linker is ever needed on the
    host.

Assumes the image already exists (see build_image.py).
"""

from __future__ import annotations

import argparse

import fetch_onnxruntime
from targets import Target, cargo_build_command, podman_exec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()

    # Not needed to *compile* -- load-dynamic links nothing at build time -- but needed to run
    # and to package, so it is fetched up front either way.
    fetch_onnxruntime.fetch(args.target)

    podman_exec(args.target, ["bash", "-c", cargo_build_command(args.target)])


if __name__ == "__main__":
    main()
