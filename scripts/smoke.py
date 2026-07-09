#!/usr/bin/env python3
"""Post-build smoke test for a target's ort_runner binary. Run automatically after each build.

Small on purpose -- it checks the things a cross-compile most easily gets silently wrong:
  * the binary is the ELF architecture the target is supposed to produce (an armv7 build that
    quietly emitted arm64, say, would pass compilation but fail here);
  * ONNX Runtime's shared library was bundled next to it (so $ORIGIN rpath resolves on-device);
  * for a Linux target, `ort_runner --version` actually runs -- which loads the bundled .so and
    calls into ONNX Runtime to read its version, proving the binary links and executes on-target.

Android binaries can't run at build time (they need a device), so those get the static checks
only; the on-device run is exercised separately by scripts/run_android.py.

The ELF checks read the header directly and inspect the bundled file on the (bind-mounted) host
filesystem, so they need no readelf/file tool and no container.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from targets import Target, resolve, run_target_binary

# ELF e_machine values (ELF header offset 18) for the architectures these targets emit.
_EM_X86_64 = 0x3E
_EM_AARCH64 = 0xB7
_EM_ARM = 0x28

_EXPECTED_MACHINE: dict[Target, int] = {
    Target.LINUX_X64: _EM_X86_64,
    Target.LINUX_ARM64: _EM_AARCH64,
    Target.ANDROID_ARM64: _EM_AARCH64,
    Target.ANDROID_ARM32: _EM_ARM,
}

# Targets whose binary the build/run environment can actually execute (Linux -- a same-arch or
# emulated build container). Android needs a device, so it gets the static checks only.
_RUNTIME_SMOKE: set[Target] = {Target.LINUX_X64, Target.LINUX_ARM64}


def _fail(message: str) -> None:
    raise SystemExit(f"smoke failed: {message}")


def _read_elf_machine(binary: Path) -> int:
    """Return the ELF header's e_machine field, or fail if `binary` is not an ELF file."""
    header = binary.read_bytes()[:20]
    if header[:4] != b"\x7fELF":
        _fail(f"{binary} is not an ELF file")
    byteorder = "little" if header[5] == 1 else "big"
    return int.from_bytes(header[18:20], byteorder)


def smoke(target: Target) -> None:
    config = resolve(target)
    bin_dir = config.build_dir / "bin"
    binary = bin_dir / "ort_runner"
    if not binary.is_file():
        _fail(f"binary not found (build the target first): {binary}")

    machine = _read_elf_machine(binary)
    expected = _EXPECTED_MACHINE[target]
    if machine != expected:
        _fail(f"{binary}: ELF machine 0x{machine:02x}, expected 0x{expected:02x} for {target}")

    bundled = list(bin_dir.glob("libonnxruntime.so*"))
    if not bundled:
        _fail(f"onnxruntime shared library not bundled next to {binary}")

    if target in _RUNTIME_SMOKE:
        run_target_binary(target, [f"{config.build_dir.name}/bin/ort_runner", "--version"])

    print(f"smoke OK: {target} (arch 0x{machine:02x}, bundled {bundled[0].name})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Target, choices=list(Target))
    args = parser.parse_args()
    smoke(args.target)


if __name__ == "__main__":
    main()
