#!/usr/bin/env python3
"""Fetches ONNX Runtime, configures, and builds a target inside its podman toolchain image.

Assumes the image has already been built (see build_image.py).
"""
from __future__ import annotations

import argparse
import platform
import subprocess
from pathlib import Path

from targets import REPO_ROOT, Target, resolve


def _host_integration_args() -> list[str]:
    """Bind mounts for host integrations that only apply when the host actually has them."""
    mount_args: list[str] = []

    # Bind-mounting the host's own CA trust store and clock/timezone data lets curl (fetching
    # the SDK) work on corporate machines behind an intercepting proxy with a self-signed CA.
    # Linux-only: on macOS these paths belong to the podman-machine VM, not the real host.
    if platform.system() == "Linux":
        for host_path in (Path("/etc/ssl/certs"), Path("/etc/localtime"), Path("/etc/timezone")):
            if host_path.exists():
                mount_args += ["-v", f"{host_path}:{host_path}:ro"]

    # Created up front so the very first build already gets a persistent cache; ccache reuse
    # shouldn't depend on the host happening to have this directory already.
    host_ccache_dir = Path.home() / ".ccache"
    host_ccache_dir.mkdir(parents=True, exist_ok=True)
    mount_args += ["-v", f"{host_ccache_dir}:/root/.ccache:Z", "-e", "CCACHE_DIR=/root/.ccache"]

    return mount_args


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
            *_host_integration_args(),
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
