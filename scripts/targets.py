#!/usr/bin/env python3
"""Per-target build configuration shared by the build/run entry-point scripts.

Each Target names a concrete (os, arch) pair. Adding a new target is a data change here --
one Target member plus one TargetConfig row -- not a logic change in the entry-point scripts,
which all drive off the resolved config.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Target(str, Enum):
    LINUX_X64 = "linux-x64"
    LINUX_ARM64 = "linux-aarch64"
    ANDROID_ARM64 = "android-arm64"
    ANDROID_ARM32 = "android-armv7"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TargetConfig:
    image_tag: str
    containerfile: Path
    image_platform: str
    run_platform: str | None
    cmake_preset: str
    build_dir: Path
    # `uname -m` this target's binaries produce when run natively, or None for targets that
    # never run on the build host (Android, which is driven separately over adb). Used to
    # decide whether a built binary can execute directly on the host or must go through the
    # (possibly emulated) build container -- see run_target_binary.
    native_machine: str | None


# Image/run platforms are pinned per target (not derived from the host) so podman produces the
# intended arch every time -- observed podman reuse a stale linux/amd64 layer after another
# image pulled the same base, silently emulating instead of building native. A target whose
# platform differs from the host is built/run under QEMU emulation, which podman handles
# transparently where binfmt is registered (the same mechanism the android amd64 image already
# relies on when built on an arm64 host).
_CONFIGS: dict[Target, TargetConfig] = {
    Target.LINUX_X64: TargetConfig(
        image_tag="ort-runner-builder-linux-x64",
        containerfile=REPO_ROOT / "podman" / "Containerfile.linux-x64",
        image_platform="linux/amd64",
        run_platform="linux/amd64",
        cmake_preset="linux-x64",
        build_dir=REPO_ROOT / "build-linux-x64",
        native_machine="x86_64",
    ),
    Target.LINUX_ARM64: TargetConfig(
        # Old-glibc (debian bullseye, glibc 2.31) aarch64 base, so the binary runs on any
        # newer-glibc aarch64 device (Raspberry Pi 4 on 64-bit RPi OS, etc.) without a
        # glibc-version mismatch. ONNX Runtime is the prebuilt aarch64 .so -- only ort_runner's
        # own handful of .cpp files compile here, so building under emulation on an x86 host is
        # quick despite QEMU.
        image_tag="ort-runner-builder-linux-arm64",
        containerfile=REPO_ROOT / "podman" / "Containerfile.linux-arm64",
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        cmake_preset="linux-aarch64",
        build_dir=REPO_ROOT / "build-linux-aarch64",
        native_machine="aarch64",
    ),
    Target.ANDROID_ARM64: TargetConfig(
        # Always linux/amd64: the NDK only ships a Linux host toolchain for x86_64, regardless
        # of the arm64-v8a Android target ABI, so this also has to run under QEMU emulation on
        # an arm64 host/Podman VM (e.g. Apple Silicon).
        image_tag="ort-runner-builder-android",
        containerfile=REPO_ROOT / "podman" / "Containerfile.android",
        image_platform="linux/amd64",
        run_platform="linux/amd64",
        cmake_preset="android-arm64",
        build_dir=REPO_ROOT / "build-android-arm64",
        native_machine=None,
    ),
    Target.ANDROID_ARM32: TargetConfig(
        # Same amd64 NDK host toolchain as arm64; only the target ABI (armeabi-v7a, hardfloat)
        # differs. Shares the android builder image.
        image_tag="ort-runner-builder-android",
        containerfile=REPO_ROOT / "podman" / "Containerfile.android",
        image_platform="linux/amd64",
        run_platform="linux/amd64",
        cmake_preset="android-armeabi-v7a",
        build_dir=REPO_ROOT / "build-android-armv7",
        native_machine=None,
    ),
}


def resolve(target: Target) -> TargetConfig:
    return _CONFIGS[target]


def podman_exec(target: Target, command: list[str]) -> None:
    """Runs `command` inside `target`'s already-built podman toolchain image, repo mounted at
    /workspace. Lets build outputs run on hosts (e.g. macOS) that can't execute the target's
    binaries directly -- the build image always can, since it's what built them."""
    config = resolve(target)
    platform_args = ["--platform", config.run_platform] if config.run_platform else []
    subprocess.run(
        [
            "podman",
            "run",
            "--rm",
            *platform_args,
            "-v",
            f"{REPO_ROOT}:/workspace:Z",
            "-w",
            "/workspace",
            config.image_tag,
            *command,
        ],
        check=True,
    )


def run_target_binary(target: Target, command: list[str]) -> None:
    """Runs `command` (paths relative to the repo root) against a target's built binaries.

    When the host can execute the target's binaries directly -- a native Linux host of the same
    arch, where the build container shares the host kernel -- this runs them in place, skipping
    the container-start overhead on every invocation. Otherwise (macOS, or a different-arch
    Linux host such as x86 building the aarch64 target) it falls back to podman_exec, which runs
    them inside the target's own (possibly emulated) build image.
    """
    config = resolve(target)
    host_can_exec = platform.system() == "Linux" and platform.machine() == config.native_machine
    if host_can_exec:
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    else:
        podman_exec(target, command)
