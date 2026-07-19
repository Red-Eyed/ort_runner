#!/usr/bin/env python3
"""Runs a Linux ort_runner against a model.

    uv run scripts/run_linux.py --source build linux-aarch64 model.onnx --iterations 200
    uv run scripts/run_linux.py --source prebuilt linux-x64 model.onnx

`--source` decides which binary runs: one built from source, or one downloaded from a release.
It is required rather than defaulted because the two can disagree, and a benchmark attributed to
the wrong binary is worse than no benchmark.
"""

from __future__ import annotations

import argparse
import platform

from targets import REPO_ROOT, Source, Target, resolve, run_target_binary


def _reject_emulated(target: Target) -> None:
    """Refuse to run a downloaded binary this machine cannot execute natively.

    The build path may fall back to the target's own container, which is fine for checking that a
    binary works at all. Timing one that way is not: an x86_64 build under QEMU on an arm64 host
    produces latency figures that describe the emulator rather than the model, and they look
    exactly like real ones. Refusing is the correct answer for a measurement tool -- and it keeps
    the prebuilt path free of any container requirement, which is the reason to reach for it.
    """
    if platform.system() == "Linux" and platform.machine() == resolve(target).native_machine:
        return
    raise SystemExit(
        f"error: this host ({platform.system()} {platform.machine()}) cannot natively execute a "
        f"{target} binary.\n"
        "  Timing it under emulation would measure the emulator rather than the model, so it is\n"
        "  refused. Run it on a matching machine, or use `just run-dev-" + str(target) + "` to\n"
        "  exercise it in a container without trusting the numbers."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Source,
        choices=list(Source),
        required=True,
        help="run the binary built from source, or one downloaded from a release",
    )
    parser.add_argument("target", type=Target, choices=[Target.LINUX_X64, Target.LINUX_ARM64])
    parser.add_argument("model")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.source is Source.PREBUILT:
        _reject_emulated(args.target)

    binary = args.source.directory(args.target) / "ort_runner"
    if not binary.is_file():
        recipe = (
            f"download-prebuilt {args.target}"
            if args.source is Source.PREBUILT
            else f"build-{args.target}"
        )
        raise SystemExit(f"error: {binary} not found; run `just {recipe}` first")

    runner = str(binary.relative_to(REPO_ROOT))
    run_target_binary(args.target, [runner, "--model", args.model, *args.args])


if __name__ == "__main__":
    main()
