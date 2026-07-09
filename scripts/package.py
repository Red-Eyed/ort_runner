#!/usr/bin/env python3
"""Package built targets into distributable zips under dist/.

Each zip is a self-contained, runnable ort_runner: just the binary plus the ONNX Runtime shared
library bundled beside it (so the binary's $ORIGIN rpath resolves on extraction) -- no source.
The version in the zip name is read from CMakeLists.txt, the single source of the project version.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

from fetch_onnxruntime import ORT_VERSION
from targets import REPO_ROOT, Target, resolve

DIST_DIR = REPO_ROOT / "dist"


def project_version() -> str:
    text = (REPO_ROOT / "CMakeLists.txt").read_text()
    match = re.search(r"project\(\s*\S+\s+VERSION\s+(\d+\.\d+\.\d+)", text)
    if not match:
        raise SystemExit("error: could not read project VERSION from CMakeLists.txt")
    return match.group(1)


def _add_file(archive: zipfile.ZipFile, src: Path, arcname: str, executable: bool) -> None:
    """Add `src` to the zip under `arcname`, recording unix permissions (extract keeps +x)."""
    info = zipfile.ZipInfo(arcname)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if executable else 0o644) << 16
    archive.writestr(info, src.read_bytes())


def package(target: Target) -> Path:
    config = resolve(target)
    bin_dir = config.build_dir / "bin"
    binary = bin_dir / "ort_runner"
    if not binary.is_file():
        raise SystemExit(f"error: build {target} first, missing {binary}")
    libs = sorted(bin_dir.glob("libonnxruntime.so*"))
    if not libs:
        raise SystemExit(f"error: onnxruntime library not bundled in {bin_dir}")

    stem = f"ort_runner-v{project_version()}-{target}-ort{ORT_VERSION}"
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        _add_file(archive, binary, f"{stem}/ort_runner", executable=True)
        for lib in libs:
            _add_file(archive, lib, f"{stem}/{lib.name}", executable=False)
    print(zip_path)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", type=Target, choices=list(Target), nargs="+")
    args = parser.parse_args()
    for target in args.targets:
        package(target)


if __name__ == "__main__":
    main()
