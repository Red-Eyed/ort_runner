#!/usr/bin/env python3
"""Find the newest ONNX Runtime release this repo can package automatically.

The scheduled workflow uses this as the cheap gate before starting container builds. Android is
the release gate: if the Android AARs are not present, the version is skipped. Linux targets are
secondary and are included only when their expected artifacts are available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import fetch_onnxruntime
from targets import Target

UPSTREAM_RELEASES = "https://api.github.com/repos/microsoft/onnxruntime/releases?per_page=20"
UPSTREAM_DOWNLOAD = "https://github.com/microsoft/onnxruntime/releases/download"
MAVEN = "https://repo1.maven.org/maven2/com/microsoft/onnxruntime"
DEFAULT_REPOSITORY = "Red-Eyed/ort_runner"
RELEASE_TAG_PREFIX = "onnxruntime-v"


@dataclass(frozen=True)
class Candidate:
    version: str
    tag: str
    targets: tuple[Target, ...]


@dataclass(frozen=True)
class CheckResult:
    release_needed: bool
    reason: str
    candidate: Candidate | None = None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "ort-runner-onnxruntime-release",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _json(url: str) -> object:
    request = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _exists(url: str, headers: dict[str, str] | None = None) -> bool:
    request = urllib.request.Request(url, method="HEAD", headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30):
            return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def _version_key(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"not a three-part version: {version}")
    return tuple(int(part) for part in parts)


def _release_versions(releases: object) -> list[str]:
    if not isinstance(releases, list):
        raise SystemExit("error: GitHub releases response was not a list")

    versions = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        if release.get("draft") or release.get("prerelease"):
            continue

        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue

        match = re.fullmatch(r"v(\d+\.\d+\.\d+)", tag)
        if match:
            versions.append(match.group(1))
    return sorted(versions, key=_version_key, reverse=True)


def _android_artifacts_ready(version: str) -> bool:
    return _exists(_maven_aar("onnxruntime-android", version)) and _exists(
        _maven_aar("onnxruntime-android-qnn", version)
    )


def _maven_aar(artifact: str, version: str) -> str:
    return f"{MAVEN}/{artifact}/{version}/{artifact}-{version}.aar"


def _linux_targets_ready(version: str) -> tuple[Target, ...]:
    targets = []
    linux_assets = {
        Target.LINUX_X64: f"{UPSTREAM_DOWNLOAD}/v{version}/onnxruntime-linux-x64-{version}.tgz",
        Target.LINUX_ARM64: (
            f"{UPSTREAM_DOWNLOAD}/v{version}/onnxruntime-linux-aarch64-{version}.tgz"
        ),
    }
    for target, url in linux_assets.items():
        if _exists(url, headers=_github_headers()):
            targets.append(target)
    return tuple(targets)


def _release_exists(repository: str, version: str) -> bool:
    tag = f"{RELEASE_TAG_PREFIX}{version}"
    url = f"https://api.github.com/repos/{repository}/releases/tags/{tag}"
    return _exists(url, headers=_github_headers())


def check(repository: str, minimum_version: str) -> CheckResult:
    versions = _release_versions(_json(UPSTREAM_RELEASES))
    if not versions:
        return CheckResult(False, "no stable upstream ONNX Runtime releases found")

    for version in versions:
        if _version_key(version) < _version_key(minimum_version):
            continue
        if _release_exists(repository, version):
            continue
        if not _android_artifacts_ready(version):
            continue

        android_targets = (Target.ANDROID_ARM64, Target.ANDROID_ARM32, Target.ANDROID_X64)
        targets = android_targets + _linux_targets_ready(version)
        return CheckResult(
            True,
            f"ONNX Runtime {version} is ready for Android; selected {len(targets)} targets",
            Candidate(version=version, tag=f"{RELEASE_TAG_PREFIX}{version}", targets=targets),
        )

    return CheckResult(
        False,
        f"no unreleased ONNX Runtime at or above {minimum_version} has Android artifacts ready",
    )


def _write_github_outputs(path: Path, result: CheckResult) -> None:
    lines = [
        f"release_needed={str(result.release_needed).lower()}",
        f"reason={result.reason}",
    ]
    if result.candidate:
        lines.extend(
            [
                f"ort_version={result.candidate.version}",
                f"tag={result.candidate.tag}",
                "targets=" + " ".join(str(target) for target in result.candidate.targets),
            ]
        )
    with path.open("a") as output:
        for line in lines:
            print(line, file=output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        help="repository whose ONNX Runtime refresh releases are checked",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=os.environ.get("GITHUB_OUTPUT"),
        help="write GitHub Actions outputs to this file",
    )
    parser.add_argument(
        "--minimum-version",
        default=fetch_onnxruntime.DEFAULT_ORT_VERSION,
        help="ignore upstream versions at or below this version",
    )
    args = parser.parse_args()

    result = check(args.repository, args.minimum_version)
    print(result.reason)
    if args.github_output:
        _write_github_outputs(args.github_output, result)


if __name__ == "__main__":
    main()
