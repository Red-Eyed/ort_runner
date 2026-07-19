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
import shutil

import fetch_onnxruntime
from targets import Target, cargo_build_command, podman_exec, resolve

# What ort_runner looks for beside its own executable (see src/dylib.rs).
BUNDLED_LIBRARY_NAME = "libonnxruntime.so"


def bundle_runtime(target: Target) -> None:
    """Copy ONNX Runtime beside the freshly built binary.

    Under `load-dynamic` nothing links against ONNX Runtime, so cargo produces a binary that will
    not run until the shared library is next to it -- which is exactly where `dylib::resolve`
    looks, and what makes an extracted release zip runnable with no LD_LIBRARY_PATH. The C++ build
    got this from a CMake install step; with cargo there is nowhere else for it to happen.
    """
    destination = resolve(target).build_dir / BUNDLED_LIBRARY_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    # copyfile, not copy2: the source is the tail of a symlink chain, and what belongs beside the
    # binary is the payload under the plain soname, not a link to a path that will not exist.
    shutil.copyfile(fetch_onnxruntime.library_path(target), destination)
    print(f"bundled {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()

    # Not needed to *compile* -- load-dynamic links nothing at build time -- but needed to run
    # and to package, so it is fetched up front either way.
    fetch_onnxruntime.fetch(args.target)

    podman_exec(args.target, ["bash", "-c", cargo_build_command(args.target)])
    bundle_runtime(args.target)


if __name__ == "__main__":
    main()
