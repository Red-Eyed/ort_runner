#!/usr/bin/env python3
"""Download a pinned prebuilt ONNX Runtime distribution into sdk/ for a target.

ort_runner never builds ONNX Runtime -- it always links the official prebuilt binaries:
  * Linux (x86_64 / aarch64): the onnxruntime-linux-<arch> release tarball (a glibc .so).
  * Android (arm64-v8a / armeabi-v7a): the shared lib from inside the onnxruntime-android AAR.

Each target unpacks into its own arch-specific sdk/ subdirectory (so the two Linux arches, or the
two Android ABIs, never clobber one another), matching the ORT_RUNNER_SDK_DIR each CMakePresets.json
preset points at. Runs inside the build container (invoked by scripts/build.py), where the download
lands in the bind-mounted repo and TLS uses the container's ca-certificates.

Uses only the standard library, so it needs no extra packages installed to run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from targets import REPO_ROOT, Target

ORT_VERSION = "1.27.0"
SDK_ROOT = REPO_ROOT / "sdk"

_GITHUB_RELEASE = f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}"
_MAVEN_AAR = (
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/"
    f"{ORT_VERSION}/onnxruntime-android-{ORT_VERSION}.aar"
)


@dataclass(frozen=True)
class TarballDist:
    """An onnxruntime-linux-<arch> release tarball: flat include/ + lib/libonnxruntime.so*."""

    url: str
    dest: Path

    def already_present(self) -> bool:
        return (self.dest / "include" / "onnxruntime_cxx_api.h").is_file()


@dataclass(frozen=True)
class AarDist:
    """The onnxruntime-android AAR (a zip): headers/ + jni/<abi>/libonnxruntime.so per ABI."""

    url: str
    dest: Path
    abi: str

    def already_present(self) -> bool:
        return (self.dest / "headers" / "onnxruntime_cxx_api.h").is_file()


_DISTS: dict[Target, TarballDist | AarDist] = {
    Target.LINUX_X64: TarballDist(
        url=f"{_GITHUB_RELEASE}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
        dest=SDK_ROOT / "onnxruntime-linux-x64",
    ),
    Target.LINUX_ARM64: TarballDist(
        url=f"{_GITHUB_RELEASE}/onnxruntime-linux-aarch64-{ORT_VERSION}.tgz",
        dest=SDK_ROOT / "onnxruntime-linux-aarch64",
    ),
    Target.ANDROID_ARM64: AarDist(
        url=_MAVEN_AAR,
        dest=SDK_ROOT / "onnxruntime-android-arm64",
        abi="arm64-v8a",
    ),
    Target.ANDROID_ARM32: AarDist(
        url=_MAVEN_AAR,
        dest=SDK_ROOT / "onnxruntime-android-armeabi-v7a",
        abi="armeabi-v7a",
    ),
}


def _download(url: str, dest_file: Path) -> None:
    print(f"Downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url) as response, dest_file.open("wb") as out:
        shutil.copyfileobj(response, out)


def _extract_tarball_strip1(archive: Path, dest: Path) -> None:
    """Unpack `archive` into `dest`, dropping the tarball's single top-level directory.

    The tarball's top entry is named after the asset (e.g. onnxruntime-linux-aarch64-1.27.0/),
    not the fixed dest, so strip that first path component to land include/ and lib/ in dest.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        members = []
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            member.name = str(Path(*parts[1:]))
            members.append(member)
        tar.extractall(dest, members=members)


def _extract_aar_abi(archive: Path, dest: Path, abi: str) -> None:
    """Unpack only headers/ and jni/<abi>/libonnxruntime.so from the AAR zip into `dest`.

    The Java/JNI glue lib and classes.jar in the AAR are irrelevant to a native CLI, and each
    target extracts a single ABI so its dest holds exactly one libonnxruntime.so.
    """
    dest.mkdir(parents=True, exist_ok=True)
    abi_lib = f"jni/{abi}/libonnxruntime.so"
    with zipfile.ZipFile(archive) as aar:
        wanted = [n for n in aar.namelist() if n.startswith("headers/") or n == abi_lib]
        if abi_lib not in wanted:
            raise SystemExit(f"error: {abi_lib} not found in AAR {archive.name}")
        aar.extractall(dest, members=wanted)


def fetch(target: Target) -> Path:
    """Ensure `target`'s ONNX Runtime SDK is unpacked under sdk/; return its directory."""
    dist = _DISTS[target]
    if dist.already_present():
        print(f"Already present: {dist.dest}", file=sys.stderr)
        return dist.dest

    SDK_ROOT.mkdir(parents=True, exist_ok=True)
    archive = SDK_ROOT / Path(dist.url).name
    _download(dist.url, archive)
    if isinstance(dist, TarballDist):
        _extract_tarball_strip1(archive, dist.dest)
    else:
        _extract_aar_abi(archive, dist.dest, dist.abi)
    archive.unlink()
    return dist.dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()
    print(f"ORT_RUNNER_SDK_DIR={fetch(args.target)}")


if __name__ == "__main__":
    main()
