#!/usr/bin/env python3
"""Publish a GitHub release for the current project version.

Packages every target into dist/ (see package.py), reads the release notes from the matching
CHANGELOG.md section, pushes the current branch, and creates the GitHub release -- tagged
v<version> with the per-target zips attached -- via the gh CLI.

Assumes the version/changelog commit is already made and every target is built; run
`just release`, which builds all targets first, to do the whole thing from scratch.
"""

from __future__ import annotations

import argparse
import subprocess

import package
from targets import REPO_ROOT, Target


def changelog_notes(version: str) -> str:
    """Return the body of CHANGELOG.md's `## [<version>]` section (its release notes)."""
    lines = (REPO_ROOT / "CHANGELOG.md").read_text().splitlines()
    heading = f"## [{version}]"
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    except StopIteration:
        raise SystemExit(f"error: no '{heading}' section in CHANGELOG.md") from None
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("## ["):
            break
        body.append(line)
    return "\n".join(body).strip()


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def publish(dry_run: bool) -> None:
    version = package.project_version()
    tag = f"v{version}"
    zips = [package.package(target) for target in Target]
    notes = changelog_notes(version)
    branch = current_branch()

    gh_command = [
        "gh",
        "release",
        "create",
        tag,
        "--title",
        tag,
        "--notes",
        notes,
        "--target",
        branch,
        *[str(zip_path) for zip_path in zips],
    ]
    if dry_run:
        print("dry run -- would push and publish:")
        print(f"+ git push origin {branch}")
        print("+ " + " ".join(gh_command))
        return

    _run(["git", "push", "origin", branch])
    _run(gh_command)
    print(f"published {tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="package and print what would be pushed/published, without doing it",
    )
    args = parser.parse_args()
    publish(args.dry_run)


if __name__ == "__main__":
    main()
