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
import json
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from targets import REPO_ROOT, Target, resolve

REPO = "Red-Eyed/ort_runner"
API = f"https://api.github.com/repos/{REPO}/releases"

# GitHub rejects API requests with no User-Agent.
HEADERS = {"User-Agent": "ort_runner-download-prebuilt", "Accept": "application/vnd.github+json"}

DOWNLOAD_CACHE = REPO_ROOT / "prebuilt" / ".zips"


def _get_json(url: str) -> dict:
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


def resolve_release(version: str) -> dict:
    """The release payload for `version`, or the latest release when given "latest"."""
    if version == "latest":
        return _get_json(f"{API}/latest")
    tag = version if version.startswith("v") else f"v{version}"
    return _get_json(f"{API}/tags/{tag}")


def asset_for(release: dict, target: Target) -> dict:
    """The release asset belonging to `target`.

    Matched on the target name delimited by dashes rather than by rebuilding the full asset
    name: the name also carries the ONNX Runtime version, which moves independently, so an exact
    match would break every time the runtime is bumped.
    """
    marker = f"-{target}-"
    matches = [asset for asset in release.get("assets", []) if marker in asset["name"]]
    if not matches:
        names = sorted(asset["name"] for asset in release.get("assets", []))
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
    release = resolve_release(version)
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
