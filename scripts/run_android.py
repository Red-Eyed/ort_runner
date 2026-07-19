#!/usr/bin/env python3
"""Pushes the Android build + libonnxruntime.so to a connected device/emulator and runs it there."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

from targets import Source, Target

DEVICE_DIR = "/data/local/tmp"


def require_file(path: Path, what: str) -> Path:
    """Return path if it is an existing file; otherwise abort with a clear message."""
    if not path.is_file():
        raise SystemExit(f"error: {what} not found: {path}")
    return path


def push(local: Path, remote: str) -> None:
    subprocess.run(["adb", "push", str(local), remote], check=True)


def push_host_file(local: Path) -> str:
    """Push a host file into DEVICE_DIR and return its device-side path."""
    remote = f"{DEVICE_DIR}/{local.name}"
    push(local, remote)
    return remote


def to_device_arg(arg: str) -> str:
    """Rewrite an arg naming an existing host file to its pushed device path; leave others as-is."""
    path = Path(arg)
    if path.is_file():
        return push_host_file(path)
    return arg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Source,
        choices=list(Source),
        required=True,
        help="push the binary built from source, or one downloaded from a release",
    )
    parser.add_argument(
        "target",
        type=Target,
        choices=[Target.ANDROID_ARM64, Target.ANDROID_ARM32, Target.ANDROID_X64],
    )
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if shutil.which("adb") is None:
        raise SystemExit("error: adb not found on PATH; install Android platform-tools")

    bin_dir = args.source.directory(args.target)
    recipe = (
        f"download-prebuilt {args.target}"
        if args.source is Source.PREBUILT
        else f"build-{args.target}"
    )
    runner = require_file(bin_dir / "ort_runner", f"ort_runner binary (run `just {recipe}` first)")
    libort = require_file(
        bin_dir / "libonnxruntime.so",
        f"libonnxruntime.so beside it in {bin_dir}",
    )
    model = require_file(Path(args.model), "model")

    push(runner, f"{DEVICE_DIR}/ort_runner")
    push(libort, f"{DEVICE_DIR}/libonnxruntime.so")
    subprocess.run(["adb", "shell", "chmod", "+x", f"{DEVICE_DIR}/ort_runner"], check=True)

    device_model = push_host_file(model)
    device_args = [to_device_arg(arg) for arg in args.args]

    # Android's linker ignores the binary's $ORIGIN RUNPATH, so libonnxruntime.so
    # (pushed alongside ort_runner in DEVICE_DIR) is only found via LD_LIBRARY_PATH.
    command = [f"{DEVICE_DIR}/ort_runner", "--model", device_model, *device_args]
    remote_command = f"LD_LIBRARY_PATH={DEVICE_DIR} {shlex.join(command)}"
    subprocess.run(["adb", "shell", remote_command], check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"error: command failed (exit {exc.returncode}): {shlex.join(exc.cmd)}")
