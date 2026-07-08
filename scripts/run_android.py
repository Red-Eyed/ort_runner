#!/usr/bin/env python3
"""Pushes the Android build + libonnxruntime.so to a connected device/emulator and runs it there."""
from __future__ import annotations

import argparse
import subprocess

from targets import REPO_ROOT

DEVICE_DIR = "/data/local/tmp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    build_dir = REPO_ROOT / "build-android"
    subprocess.run(
        ["adb", "push", str(build_dir / "bin" / "ort_runner"), f"{DEVICE_DIR}/ort_runner"],
        check=True,
    )
    subprocess.run(
        ["adb", "push", str(build_dir / "bin" / "libonnxruntime.so"), f"{DEVICE_DIR}/libonnxruntime.so"],
        check=True,
    )
    subprocess.run(["adb", "shell", "chmod", "+x", f"{DEVICE_DIR}/ort_runner"], check=True)

    remote_command = (
        f"LD_LIBRARY_PATH={DEVICE_DIR} {DEVICE_DIR}/ort_runner --model {args.model} {' '.join(args.args)}"
    )
    subprocess.run(["adb", "shell", remote_command], check=True)


if __name__ == "__main__":
    main()
