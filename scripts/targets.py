#!/usr/bin/env python3
"""Per-target build configuration shared by the build/run entry-point scripts.

Each Target names a concrete (os, arch) pair. Adding a target is a data change here -- one
Target member plus one TargetConfig row -- not a logic change in the entry-point scripts, which
all drive off the resolved config.
"""

from __future__ import annotations

import os
import platform
import subprocess
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# No linux-armv7 target: Microsoft publishes ONNX Runtime tarballs for linux-x64 and
# linux-aarch64 only. With no prebuilt shared library for 32-bit ARM Linux there is nothing for
# ort_runner to load, and building ONNX Runtime from source is out of scope. 32-bit ARM is
# still covered on Android, where the AAR does ship an armeabi-v7a library.


class Target(str, Enum):
    LINUX_X64 = "linux-x64"
    LINUX_ARM64 = "linux-aarch64"
    ANDROID_ARM64 = "android-arm64"
    ANDROID_ARM32 = "android-armv7"
    ANDROID_X64 = "android-x86_64"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TargetConfig:
    image_tag: str
    containerfile: Path
    image_platform: str
    run_platform: str | None
    # Rust target triple. Replaces the CMake preset the C++ build selected.
    rust_triple: str
    # Android ABI name, used when packaging and when pushing to a device; None for Linux.
    android_abi: str | None
    # `uname -m` this target's binaries produce when run natively, or None for targets that
    # never run on the build host (Android, driven separately over adb). Decides whether a built
    # binary can execute directly on the host or must go through the build container.
    native_machine: str | None

    @property
    def host_arch(self) -> str:
        """Architecture of the container that compiles this target ("amd64" / "arm64")."""
        return self.image_platform.split("/")[-1]

    @property
    def target_dir(self) -> Path:
        """CARGO_TARGET_DIR for this target, on the host.

        Keyed by the build *image*, not by architecture and not by target triple.

        Cargo namespaces target artifacts per triple, but build scripts and proc macros are
        compiled for the host and land in a shared target/release/ with nothing distinguishing
        them. Architecture alone is too coarse a key: the Linux image (Ubuntu 18.04, glibc 2.27)
        and the Android image (Debian 12, glibc 2.36) are both arm64, so sharing a directory
        made Ubuntu try to execute build scripts linked against glibc 2.36 -- "failed to run
        custom build command for libc". One directory per image is the only key that holds.
        """
        return REPO_ROOT / "target" / self.image_tag

    @property
    def build_dir(self) -> Path:
        """Where cargo places this target's finished binaries."""
        return self.target_dir / self.rust_triple / "release"


# Image/run platforms are pinned per target (not derived from the host) so podman produces the
# intended architecture every time -- podman has been observed reusing a stale linux/amd64 layer
# after another image pulled the same base, silently emulating instead of building native. A
# target whose platform differs from the host builds under QEMU emulation, which podman handles
# transparently where binfmt is registered.
_CONFIGS: dict[Target, TargetConfig] = {
    Target.LINUX_X64: TargetConfig(
        image_tag="ort-runner-builder-linux",
        containerfile=REPO_ROOT / "podman" / "Containerfile.linux",
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        rust_triple="x86_64-unknown-linux-gnu",
        android_abi=None,
        native_machine="x86_64",
    ),
    Target.LINUX_ARM64: TargetConfig(
        # glibc 2.27 matches ONNX Runtime's own floor exactly, so the binary runs anywhere
        # ONNX Runtime does -- including Raspberry Pi OS Buster (2.28).
        image_tag="ort-runner-builder-linux",
        containerfile=REPO_ROOT / "podman" / "Containerfile.linux",
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        rust_triple="aarch64-unknown-linux-gnu",
        android_abi=None,
        native_machine="aarch64",
    ),
    Target.ANDROID_ARM64: TargetConfig(
        # Always linux/amd64: the NDK ships a Linux host toolchain for x86_64 only, regardless
        # of the arm64-v8a target ABI, so this runs under QEMU emulation on an arm64 host.
        image_tag="ort-runner-builder-android",
        containerfile=REPO_ROOT / "podman" / "Containerfile.android",
        # Android binaries never run on the build host; they go to a device over adb.
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        rust_triple="aarch64-linux-android",
        android_abi="arm64-v8a",
        native_machine=None,
    ),
    Target.ANDROID_ARM32: TargetConfig(
        # Same amd64 NDK host toolchain as arm64; only the target ABI differs. Shares the image.
        image_tag="ort-runner-builder-android",
        containerfile=REPO_ROOT / "podman" / "Containerfile.android",
        # Android binaries never run on the build host; they go to a device over adb.
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        rust_triple="armv7-linux-androideabi",
        android_abi="armeabi-v7a",
        native_machine=None,
    ),
    Target.ANDROID_X64: TargetConfig(
        # The emulator ABI. Shares the Android image; only the linker flags differ.
        image_tag="ort-runner-builder-android",
        containerfile=REPO_ROOT / "podman" / "Containerfile.android",
        image_platform="linux/arm64",
        run_platform="linux/arm64",
        rust_triple="x86_64-linux-android",
        android_abi="x86_64",
        native_machine=None,
    ),
}


def resolve(target: Target) -> TargetConfig:
    return _CONFIGS[target]


# Cargo's network path hangs indefinitely inside these containers (0% CPU, blocked on a
# socket) even though curl reaches index.crates.io fine. --offline sidesteps it entirely and is
# correct anyway: every dependency is already vendored in the bind-mounted .cargo/registry, and
# a build that silently reaches the network is not hermetic. Use `just fetch` when Cargo.toml
# changes and new crates are genuinely needed.
OFFLINE = "--offline"


def cargo_build_command(target: Target) -> str:
    """The cargo invocation that builds `target`, run inside its toolchain image.

    Identical for every target: the Android images point cargo at the NDK's clang wrappers
    through CARGO_TARGET_<TRIPLE>_LINKER, so no per-ABI wrapper tool is involved, and
    `load-dynamic` means nothing links against ONNX Runtime anywhere.
    """
    return f"cargo build {OFFLINE} --release --target {resolve(target).rust_triple}"


# No containerised run may hang forever. A silent hang is this repo's worst failure mode: it
# looks identical to a slow cross-compile, so it gets waited on rather than investigated, and it
# has burned entire sessions. Anything still running past this is not making progress -- a cold
# cross-compile of the whole dependency graph finishes well inside it.
DEFAULT_TIMEOUT_SECONDS = 1800


def _kill_container(name: str) -> None:
    """Stops a container the timeout gave up on.

    Killing the podman *client* does not stop the container -- it keeps running detached, holding
    memory and the CARGO_TARGET_DIR lock, so the next run inherits the mess. That is why the run
    below is named: the name is the only handle left once the client is gone.
    """
    subprocess.run(["podman", "kill", name], check=False, capture_output=True)


def podman_exec(
    target: Target, command: list[str], timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
) -> None:
    """Runs `command` inside `target`'s toolchain image, repo mounted at /workspace.

    The same image builds and runs: its glibc 2.27 is low enough that produced binaries stay
    portable, and high enough to dlopen libonnxruntime.so. So this serves both the build and
    running built binaries on hosts (macOS) that cannot execute them directly.

    Bounded by `timeout_seconds`: a run that exceeds it is killed and raises, so a hang always
    surfaces as a failure with a diagnostic rather than an indefinite wait. Every containerised
    command in this repo goes through here, which is what makes that guarantee unavoidable
    rather than something each caller has to remember.
    """
    config = resolve(target)
    image = config.image_tag
    platform_args = ["--platform", config.run_platform] if config.run_platform else []
    # Set here rather than in the image: it depends on the container's architecture, and one
    # Containerfile serves both Linux arches.
    target_dir_args = ["-e", f"CARGO_TARGET_DIR=/workspace/target/{config.image_tag}"]
    name = f"ort-runner-{target.value}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(
            [
                "podman",
                "run",
                "--rm",
                "--name",
                name,
                *platform_args,
                *target_dir_args,
                "-v",
                f"{REPO_ROOT}:/workspace:Z",
                "-w",
                "/workspace",
                image,
                *command,
            ],
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        _kill_container(name)
        raise SystemExit(
            f"timed out after {timeout_seconds}s and was killed:\n"
            f"    {' '.join(command)}\n"
            f"This is a hang, not slowness. Check what the container was blocked on with\n"
            f"`podman ps` and `podman top <id>` while it runs -- a test binary sitting in\n"
            f"futex_do_wait means it reached ONNX Runtime without a loaded library."
        ) from None


def run_target_binary(target: Target, command: list[str]) -> None:
    """Runs `command` (paths relative to the repo root) against a target's built binaries.

    When the host can execute the target's binaries directly -- a native Linux host of the same
    arch, where the build container shares the host kernel -- this runs them in place, skipping
    container-start overhead. Otherwise (macOS, or a different-arch Linux host) it falls back to
    podman_exec, which runs them inside the target's own build image.
    """
    config = resolve(target)
    host_can_exec = platform.system() == "Linux" and platform.machine() == config.native_machine
    if host_can_exec:
        subprocess.run(command, check=True, cwd=REPO_ROOT)
    else:
        podman_exec(target, command)
