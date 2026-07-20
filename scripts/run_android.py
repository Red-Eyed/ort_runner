#!/usr/bin/env python3
"""Pushes the Android build + libonnxruntime.so to a connected device/emulator and runs it there."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from targets import Source, Target

DEVICE_DIR = "/data/local/tmp"

# Where a QAIRT/QNN SDK install announces itself. Qualcomm's own scripts set it, so an
# already-configured machine needs no flag.
QNN_SDK_ENV = "QNN_SDK_ROOT"


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


def device_dir_exists(remote: str) -> bool:
    probe = subprocess.run(["adb", "shell", "test", "-d", remote], check=False)
    return probe.returncode == 0


def pull_artifacts(bin_dir: Path) -> None:
    """Bring the run's output directories back from the device.

    ort_runner anchors reports/ and ort_profiler/ to its own location rather than to the working
    directory, which on a device means DEVICE_DIR. Pulling them beside the host-side binary
    reproduces the layout a local run leaves behind, so a result is found the same way whether it
    was measured here or on a phone.

    Everything is pulled, not just this run's output: a report is only worth keeping because
    getting a device back into the state that produced it is the expensive part, and that argues
    for retrieving whatever is there rather than reasoning about which file is new.
    """
    for name in ("reports", "ort_profiler"):
        remote = f"{DEVICE_DIR}/{name}"
        # ort_profiler/ only exists when --profile asked for it, and a missing directory is the
        # normal case rather than a failure.
        if not device_dir_exists(remote):
            continue
        subprocess.run(["adb", "pull", remote, str(bin_dir)], check=True)
        print(f"pulled {name}/ to {bin_dir / name}")


def to_device_arg(arg: str) -> str:
    """Rewrite an arg naming an existing host file to its pushed device path; leave others as-is."""
    path = Path(arg)
    if path.is_file():
        return push_host_file(path)
    return arg


def wants_qnn(args: list[str]) -> bool:
    """Whether this run asked for the QNN provider.

    The backend libraries below are only fetched and pushed for a run that needs them: they are
    tens of megabytes, and every other provider runs without them.
    """
    pairs = zip(args, args[1:])
    return "--provider=qnn" in args or any(a == "--provider" and b == "qnn" for a, b in pairs)


def qnn_library_dirs(explicit: Path | None) -> list[Path]:
    """Directories holding the QNN backend libraries this device may need.

    A QAIRT SDK splits them in two: `lib/aarch64-android` holds what the CPU side loads
    (libQnnHtp.so, libQnnSystem.so and the per-DSP stubs), while each `lib/hexagon-vNN/unsigned`
    holds the skel that runs on the DSP itself. Which hexagon version a phone wants follows from
    its SoC, so every version present is offered rather than one being guessed at.
    """
    if explicit is not None:
        return [require_dir(explicit, "--qnn-libs directory")]

    sdk_root = os.environ.get(QNN_SDK_ENV)
    if sdk_root is None:
        raise SystemExit(
            "error: --provider qnn needs Qualcomm's QNN backend libraries, which cannot be "
            "redistributed and so are not bundled.\n"
            f"       Set {QNN_SDK_ENV} to a QAIRT SDK install, or pass --qnn-libs <dir> holding "
            "libQnnHtp.so, libQnnSystem.so and the matching libQnnHtpV*Stub/Skel libraries."
        )

    lib_root = require_dir(Path(sdk_root) / "lib", f"{QNN_SDK_ENV}/lib")
    dirs = [d for d in [lib_root / "aarch64-android"] if d.is_dir()]
    dirs += sorted(d for d in lib_root.glob("hexagon-v*/unsigned") if d.is_dir())
    if not dirs:
        raise SystemExit(f"error: no QNN backend libraries found under {lib_root}")
    return dirs


def push_qnn_libraries(dirs: list[Path]) -> None:
    """Push every .so from `dirs` into DEVICE_DIR, where both linkers will look for them."""
    libraries = sorted({lib for directory in dirs for lib in directory.glob("*.so")})
    if not libraries:
        listed = ", ".join(str(d) for d in dirs)
        raise SystemExit(f"error: no .so files found in {listed}")
    for lib in libraries:
        push(lib, f"{DEVICE_DIR}/{lib.name}")


def require_dir(path: Path, what: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"error: {what} not found: {path}")
    return path


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
    parser.add_argument(
        "--qnn-libs",
        type=Path,
        help=(
            "directory holding Qualcomm's QNN backend libraries, pushed alongside the binary for "
            f"a --provider qnn run (default: derived from ${QNN_SDK_ENV})"
        ),
    )
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

    # Resolved before anything else is pushed, so a run that cannot get its backend libraries
    # says so immediately rather than after copying a model to the device.
    if wants_qnn(args.args):
        push_qnn_libraries(qnn_library_dirs(args.qnn_libs))

    device_model = push_host_file(model)
    device_args = [to_device_arg(arg) for arg in args.args]

    # Android's linker ignores the binary's $ORIGIN RUNPATH, so libonnxruntime.so
    # (pushed alongside ort_runner in DEVICE_DIR) is only found via LD_LIBRARY_PATH.
    #
    # ADSP_LIBRARY_PATH is the same idea for the other processor: the QNN skel libraries are
    # loaded by the Hexagon DSP, which does not consult LD_LIBRARY_PATH at all. Setting it
    # unconditionally costs a non-QNN run nothing.
    command = [f"{DEVICE_DIR}/ort_runner", "--model", device_model, *device_args]
    environment = f"LD_LIBRARY_PATH={DEVICE_DIR} ADSP_LIBRARY_PATH={DEVICE_DIR}"
    remote_command = f"{environment} {shlex.join(command)}"

    # Pulled even when the run fails: a benchmark that got far enough to write a profile, or that
    # died after earlier runs left reports behind, should not strand them on the device.
    try:
        subprocess.run(["adb", "shell", remote_command], check=True)
    finally:
        pull_artifacts(bin_dir)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"error: command failed (exit {exc.returncode}): {shlex.join(exc.cmd)}")
