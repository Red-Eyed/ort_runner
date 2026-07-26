#!/usr/bin/env python3
"""Publish a GitHub release for an ONNX Runtime refresh build."""

from __future__ import annotations

import argparse
import subprocess

import fetch_onnxruntime
import package
from check_onnxruntime_release import RELEASE_TAG_PREFIX
from targets import REPO_ROOT, Target


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def _target_ref() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _notes(version: str, targets: list[Target]) -> str:
    target_list = "\n".join(f"- `{target}`" for target in targets)
    project_version = package.project_version()
    return f"""Automated ONNX Runtime refresh release.

- ort_runner version: `{project_version}`
- ONNX Runtime version: `{version}`
- Upstream release: https://github.com/microsoft/onnxruntime/releases/tag/v{version}

Packaged targets:
{target_list}
"""


def publish(targets: list[Target], dry_run: bool) -> None:
    version = fetch_onnxruntime.ORT_VERSION
    tag = f"{RELEASE_TAG_PREFIX}{version}"
    zips = [package.package(target) for target in targets]
    command = [
        "gh",
        "release",
        "create",
        tag,
        "--title",
        f"ort_runner {package.project_version()} with ONNX Runtime {version}",
        "--notes",
        _notes(version, targets),
        "--target",
        _target_ref(),
        *[str(zip_path) for zip_path in zips],
    ]

    if dry_run:
        print("dry run -- would publish:")
        print("+ " + " ".join(command))
        return

    _run(command)
    print(f"published {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Target, choices=list(Target), nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    publish(args.targets, args.dry_run)


if __name__ == "__main__":
    main()
