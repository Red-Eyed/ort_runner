# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-09

### Added

- **Linux aarch64 target** (`linux-aarch64`), built for portability to devices like the
  Raspberry Pi 4: an old-glibc (debian bullseye, glibc 2.31) base plus statically linked
  libstdc++/libgcc, so the binary runs on any newer-glibc aarch64 system with only glibc as a
  runtime dependency.
- **Android armeabi-v7a target** (`android-armv7`, hardfloat), alongside the existing arm64-v8a
  build; both ABIs come from the official ONNX Runtime AAR.
- **Post-build smoke test** (`scripts/smoke.py`), run automatically after every build: verifies
  the binary's ELF architecture, that `libonnxruntime.so*` is bundled beside it, and (for Linux)
  that `ort_runner --version` runs and links against ONNX Runtime.

### Changed

- Build targets are now arch-explicit: `linux` → `linux-x64`, `android` → `android-arm64`, with
  `linux-aarch64` and `android-armv7` added. Each builds into its own `build-<target>/` directory
  and fetches into its own `sdk/onnxruntime-<target>/` directory (the two Linux arches and two
  Android ABIs no longer collide).
- Rewrote `scripts/fetch_onnxruntime.sh` as `scripts/fetch_onnxruntime.py` (standard library
  only). ONNX Runtime is always the official prebuilt binary — never built from source.
- Toolchain images obtain cmake and ninja from PyPI (in an isolated venv) so every target uses
  the same modern cmake regardless of the base distro's package age.

[0.2.0]: https://github.com/Red-Eyed/ort_runner/releases/tag/v0.2.0
