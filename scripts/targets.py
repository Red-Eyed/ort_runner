#!/usr/bin/env python3
"""Per-target build configuration shared by the build/run entry-point scripts."""
from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class Target(str, Enum):
    LINUX = "linux"
    ANDROID = "android"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class TargetConfig:
    image_tag: str
    containerfile: Path
    image_platform: str
    run_platform: str | None
    fetch_arg: str
    cmake_preset: str
    build_dir: Path


def _host_image_platform() -> str:
    # Explicit (derived from the host arch) rather than left to podman's default resolution --
    # observed podman pick up a stale linux/amd64 layer here after the linux/amd64 android
    # image pulled the same debian:12-slim base, silently producing an emulated (not native)
    # image on an arm64 host.
    arch = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine()]
    return f"linux/{arch}"


def resolve(target: Target) -> TargetConfig:
    if target is Target.LINUX:
        return TargetConfig(
            image_tag="ort-runner-builder-linux",
            containerfile=REPO_ROOT / "podman" / "Containerfile.linux",
            image_platform=_host_image_platform(),
            run_platform=None,
            fetch_arg="linux",
            cmake_preset="linux",
            build_dir=REPO_ROOT / "build-linux",
        )
    if target is Target.ANDROID:
        # Always linux/amd64: the NDK only ships a Linux host toolchain for x86_64, regardless
        # of the arm64-v8a Android target ABI, so this also has to run under QEMU emulation on
        # an arm64 host/Podman VM (e.g. Apple Silicon).
        return TargetConfig(
            image_tag="ort-runner-builder-android",
            containerfile=REPO_ROOT / "podman" / "Containerfile.android",
            image_platform="linux/amd64",
            run_platform="linux/amd64",
            fetch_arg="android",
            cmake_preset="android-arm64",
            build_dir=REPO_ROOT / "build-android",
        )
    raise ValueError(f"Unknown target: {target}")
