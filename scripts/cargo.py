#!/usr/bin/env python3
"""Runs an arbitrary cargo command for a target inside its podman toolchain image.

Backs `just test`, `just lint` and `just fmt`. Exists so those never fall back to a host
toolchain: the container is the only place cargo runs, and it reuses the same bind-mounted
CARGO_HOME and CARGO_TARGET_DIR as the build, so a test run shares compiled dependencies with
the release build instead of starting cold.

    scripts/cargo.py linux-aarch64 clippy --all-targets
"""

from __future__ import annotations

import argparse

from targets import Target, podman_exec, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="cargo subcommand and its flags")
    args = parser.parse_args()

    if not args.args:
        parser.error("expected a cargo subcommand, e.g. `test` or `clippy --all-targets`")

    config = resolve(args.target)
    # Explicit --target keeps artifacts in the same per-triple directory the release build
    # uses, so the two share compiled dependencies rather than each keeping their own copy.
    command = ["cargo", args.args[0], "--target", config.rust_triple, *args.args[1:]]
    podman_exec(args.target, ["bash", "-c", " ".join(command)])


if __name__ == "__main__":
    main()
