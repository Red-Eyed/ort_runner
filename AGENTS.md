# Project Instructions

## Batteries-Included Android Releases

When adding or maintaining Android execution-provider support, prefer release artifacts that work
without manual device setup. If a runtime dependency can legally and reliably be fetched from a
public artifact repository, package it into the Android release zip and have the run scripts copy
it to the device alongside `ort_runner` and `libonnxruntime.so`.

For QNN specifically, the intended user experience is batteries included for `android-arm64`:
ship the Qualcomm QNN runtime libraries in the release artifact when available, copy them to the
device automatically, and keep `QNN_SDK_ROOT` / `--qnn-libs` only as an override or fallback for
nonstandard builds.

Still keep hardware checks honest. QNN should remain Snapdragon-only, and unsupported devices
should get a clear message that points them to `nnapi` rather than asking them to hunt for
libraries that cannot make their hardware support QNN.
