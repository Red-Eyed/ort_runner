#!/usr/bin/env python3
"""Runs the integration tests that need a real ONNX Runtime.

`just test` deliberately runs with no runtime present, so the tests that build an actual
ort Tensor are marked #[ignore] and run from here instead. This fetches the target's SDK if it
is missing and passes the library path in, so the split costs the caller nothing:

    uv run scripts/test_e2e.py linux-aarch64
"""

from __future__ import annotations

import argparse

from fetch_onnxruntime import fetch
from targets import OFFLINE, REPO_ROOT, Target, podman_exec, resolve

# The container path of the bind-mounted repo, which is where fetch() lands the SDK.
WORKSPACE = "/workspace"

# Tighter than the default: these tests do no cross-compiling and the runtime is already
# downloaded, so anything past this is the ORT-init hang these tests were split out to contain.
TIMEOUT_SECONDS = 600


def dylib_in_container(target: Target) -> str:
    """Fetches the target's ONNX Runtime and returns its .so path as the container sees it."""
    sdk_dir = fetch(target)
    library = next(iter(sorted(sdk_dir.glob("**/libonnxruntime.so"))), None)
    if library is None:
        raise SystemExit(f"no libonnxruntime.so under {sdk_dir}; delete it and re-run to refetch")
    return f"{WORKSPACE}/{library.relative_to(REPO_ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Target,
        choices=[Target.LINUX_X64, Target.LINUX_ARM64],
        nargs="?",
        default=Target.LINUX_ARM64,
        help="only Linux targets can execute their own tests at build time",
    )
    args = parser.parse_args()

    config = resolve(args.target)
    command = " ".join(
        [
            f"ORT_RUNNER_TEST_DYLIB={dylib_in_container(args.target)}",
            "cargo",
            "test",
            OFFLINE,
            "--target",
            config.rust_triple,
            "--test",
            "tensor_construction",
            "--",
            "--ignored",
        ]
    )
    podman_exec(args.target, ["bash", "-c", command], timeout_seconds=TIMEOUT_SECONDS)


if __name__ == "__main__":
    main()
