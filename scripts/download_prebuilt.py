#!/usr/bin/env python3
"""Download released ort_runner binaries so a target can be run without building it.

    uv run scripts/download_prebuilt.py                       # every target, latest release
    uv run scripts/download_prebuilt.py android-arm64
    uv run scripts/download_prebuilt.py --version v0.3.0 linux-x64

Building from source needs Podman, a 4 GB toolchain image and a cross-compile; that is a
developer's cost and there is no reason to make anyone else pay it. A release zip is already
self-contained -- the binary plus its ONNX Runtime library -- so unpacking one is enough to run.
For Android that means adb and nothing else: scripts/run_android.py never touches a container.

Deliberately standard-library only. Adding `gh` would put a login and an install between someone
and their first benchmark, and these are public assets that need no authentication.
"""

from __future__ import annotations

import argparse
import functools
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from targets import REPO_ROOT, Target, resolve

REPO = "Red-Eyed/ort_runner"
API = f"https://api.github.com/repos/{REPO}/releases"

# GitHub rejects API requests with no User-Agent.
HEADERS = {"User-Agent": "ort_runner-download-prebuilt", "Accept": "application/vnd.github+json"}

DOWNLOAD_CACHE = REPO_ROOT / "prebuilt" / ".zips"


class ReleaseAsset(TypedDict):
    name: str
    browser_download_url: str


class Release(TypedDict):
    tag_name: str
    draft: bool
    prerelease: bool
    assets: list[ReleaseAsset]


def _get_json(url: str) -> object:
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise SystemExit(f"error: no such release ({url})") from None
        if error.code == 403:
            raise SystemExit(
                "error: GitHub API rate limit reached (60 requests/hour for anonymous use). "
                "Wait, or pass --version <tag> to skip the 'latest' lookup."
            ) from None
        raise SystemExit(f"error: GitHub API request failed: {error}") from None


def _asset_from_json(value: object) -> ReleaseAsset | None:
    if not isinstance(value, Mapping):
        return None

    name = value.get("name")
    download_url = value.get("browser_download_url")
    if not isinstance(name, str) or not isinstance(download_url, str):
        return None

    return {"name": name, "browser_download_url": download_url}


def _release_from_json(value: object) -> Release | None:
    if not isinstance(value, Mapping):
        return None

    tag_name = value.get("tag_name")
    draft = value.get("draft")
    prerelease = value.get("prerelease")
    assets = value.get("assets")
    if (
        not isinstance(tag_name, str)
        or not isinstance(draft, bool)
        or not isinstance(prerelease, bool)
        or not isinstance(assets, list)
    ):
        return None

    release_assets = [
        asset for raw_asset in assets if (asset := _asset_from_json(raw_asset)) is not None
    ]
    return {
        "tag_name": tag_name,
        "draft": draft,
        "prerelease": prerelease,
        "assets": release_assets,
    }


@functools.lru_cache(maxsize=1)
def _all_releases() -> list[Release]:
    """Every public release GitHub returns on the first page."""
    releases = _get_json(f"{API}?per_page=100")
    if not isinstance(releases, list):
        raise SystemExit("error: GitHub releases response was not a list")
    return [
        release
        for raw_release in releases
        if (release := _release_from_json(raw_release)) is not None
    ]


def _version_parts(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _asset_versions(
    asset: ReleaseAsset, target: Target
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    name = asset["name"]
    prefix = "ort_runner-v"
    marker = f"-{target}-ort"
    suffix = ".zip"
    if not name.startswith(prefix) or marker not in name or not name.endswith(suffix):
        return None

    project_version, ort_version = name[len(prefix) : -len(suffix)].split(marker, maxsplit=1)
    project_parts = _version_parts(project_version)
    ort_parts = _version_parts(ort_version)
    if project_parts is None or ort_parts is None:
        return None
    return project_parts, ort_parts


def _target_assets(release: Release, target: Target) -> list[ReleaseAsset]:
    marker = f"-{target}-"
    return [asset for asset in release["assets"] if marker in asset["name"]]


def _latest_release(target: Target) -> Release:
    candidates: list[tuple[tuple[int, ...], tuple[int, ...], Release]] = []
    for release in _all_releases():
        if release["draft"] or release["prerelease"]:
            continue
        matches = _target_assets(release, target)
        if len(matches) > 1:
            asset_for(release, target)
        if not matches:
            continue
        versions = _asset_versions(matches[0], target)
        if versions is not None:
            candidates.append((*versions, release))

    if not candidates:
        raise SystemExit(f"error: no release has a usable asset for {target}")
    return max(candidates, key=lambda candidate: candidate[:2])[2]


def resolve_release(version: str, target: Target) -> Release:
    """The release payload for `version`, or the newest usable release for `target`."""
    if version == "latest":
        return _latest_release(target)
    tag = version if version.startswith(("v", "onnxruntime-v")) else f"v{version}"
    release = _release_from_json(_get_json(f"{API}/tags/{tag}"))
    if release is None:
        raise SystemExit(f"error: GitHub release response for {tag} was malformed")
    return release


def asset_for(release: Release, target: Target) -> ReleaseAsset:
    """The release asset belonging to `target`.

    Matched on the target name delimited by dashes rather than by rebuilding the full asset
    name: the name also carries the ONNX Runtime version, which moves independently, so an exact
    match would break every time the runtime is bumped.
    """
    matches = _target_assets(release, target)
    if not matches:
        names = sorted(asset["name"] for asset in release["assets"])
        available = ", ".join(names) or "none"
        raise SystemExit(
            f"error: release {release['tag_name']} has no asset for {target}\n"
            f"  available: {available}"
        )
    if len(matches) > 1:
        names = ", ".join(sorted(asset["name"] for asset in matches))
        raise SystemExit(f"error: {target} matches more than one asset in this release: {names}")
    return matches[0]


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS)
    # Streamed rather than read into memory: these are tens of megabytes and the destination is a
    # file either way.
    with urllib.request.urlopen(request) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _unpack(archive: Path, destination: Path) -> None:
    """Unpack the zip into `destination`, dropping its single top-level directory.

    The archive wraps everything in a versioned folder so it extracts tidily by hand; here the
    files are wanted directly in a per-target directory, so that component is stripped -- the
    same reason fetch_onnxruntime.py strips the tarball's top level.
    """
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            if member.is_dir():
                continue
            relative = Path(*Path(member.filename).parts[1:])
            if not relative.parts:
                continue
            out_path = destination / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(zip_file.read(member))
            # The zip records unix permissions in the high half of external_attr; the binary has
            # to come back executable or the whole point of downloading it is lost.
            mode = member.external_attr >> 16
            if mode & 0o111:
                out_path.chmod(0o755)


def download(target: Target, version: str) -> Path:
    """Ensure `target`'s released binary is unpacked; return its directory."""
    destination = resolve(target).prebuilt_dir
    release = resolve_release(version, target)
    asset = asset_for(release, target)

    stamp = destination / ".release"
    if stamp.is_file() and stamp.read_text().strip() == asset["name"]:
        print(f"Already present: {destination} ({asset['name']})", file=sys.stderr)
        return destination

    archive = DOWNLOAD_CACHE / asset["name"]
    if not archive.is_file():
        print(f"Downloading {asset['name']}...", file=sys.stderr)
        _download(asset["browser_download_url"], archive)

    _unpack(archive, destination)
    # Records which asset produced this directory, so a later run can tell an up-to-date download
    # from a stale one instead of re-fetching or, worse, silently running the wrong version.
    stamp.write_text(asset["name"] + "\n")
    print(f"Unpacked {asset['name']} -> {destination}", file=sys.stderr)
    return destination


def target_named(name: str) -> Target:
    """Parse one target name, listing the valid ones when it does not match.

    A `type=` converter rather than argparse's `choices=`, because argparse also applies `choices`
    to the empty list a variadic positional yields when nothing is named -- which rejected the
    download-everything case outright, with an "invalid choice: []" that named no real mistake.
    """
    try:
        return Target(name)
    except ValueError:
        valid = ", ".join(str(target) for target in Target)
        raise argparse.ArgumentTypeError(f"invalid target '{name}' (choose from {valid})") from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # Naming nothing downloads everything, rather than being an error corrected by a separate
    # --all flag: someone who has not chosen a target wants the whole set, and a flag whose only
    # job is to say "yes, the obvious thing" is one more way to get the command wrong.
    parser.add_argument(
        "targets",
        type=target_named,
        nargs="*",
        metavar="TARGET",
        help="targets to download (default: every target)",
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="release tag to download, e.g. v0.3.0 (default: latest)",
    )
    args = parser.parse_args()

    for target in args.targets or list(Target):
        download(target, args.version)


if __name__ == "__main__":
    main()
