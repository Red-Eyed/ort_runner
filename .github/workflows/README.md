# GitHub Actions Workflows

## `release-onnxruntime.yml`

This workflow keeps released `ort_runner` binaries current with ONNX Runtime without paying the
container build cost every day.

The important policy is Android first, Linux second:

- Android artifacts gate the release. A new ONNX Runtime version is eligible only after both
  `onnxruntime-android` and `onnxruntime-android-qnn` AARs exist on Maven Central.
- Linux artifacts are optional. If Linux packages for that ONNX Runtime version are available,
  they are included; if not, the workflow still releases Android builds.
- Refresh releases are tagged as `onnxruntime-vX.Y.Z`, separate from project releases like
  `v0.6.0`.

### Trigger

```yaml
on:
  schedule:
    - cron: "17 3 * * *"
  workflow_dispatch:
```

The scheduled run is the cheap daily check. `workflow_dispatch` lets the same workflow be run
manually after an upstream release or Maven publication.

### Owner Guard, Permissions And Concurrency

```yaml
permissions:
  contents: read

concurrency:
  group: release-onnxruntime-refresh
  cancel-in-progress: false
```

Default permissions are read-only. The release job grants `contents: write` only for the job that
creates a GitHub release and uploads assets. The concurrency group prevents two refresh releases
from racing each other.

Both jobs are guarded by the repository owner and actor:

```yaml
if: ${{ github.repository_owner == 'Red-Eyed' && github.actor == 'Red-Eyed' }}
```

GitHub may still create a visible workflow run for another actor, but job-level `if` conditions
are evaluated before a runner is assigned. Unauthorized runs therefore skip every job and do not
execute scripts or receive a useful token.

For scheduled runs, GitHub associates an `actor` with the schedule. Keep the cron line owned by
`Red-Eyed`: if another write-capable user modifies the cron schedule, GitHub can make that user
the scheduled workflow actor.

### `check` Job

```yaml
jobs:
  check:
```

The check job is intentionally small and runs before any container images are built.

```yaml
- id: check
  env:
    GITHUB_TOKEN: ${{ github.token }}
  run: uv run scripts/check_onnxruntime_release.py
```

`scripts/check_onnxruntime_release.py`:

- reads stable upstream ONNX Runtime releases from GitHub;
- ignores versions below the repo's default ONNX Runtime pin;
- skips versions this repo has already published as `onnxruntime-vX.Y.Z`;
- requires the Android and Android-QNN Maven AARs;
- adds Linux targets only when their expected GitHub release artifacts exist.

If a release is needed, the script writes GitHub Actions outputs:

- `release_needed=true`
- `ort_version`, for example `1.28.0`
- `tag`, for example `onnxruntime-v1.28.0`
- `targets`, for example `android-arm64 android-armv7 android-x86_64 linux-x64`

### `release` Job Gate

```yaml
release:
  needs: check
  if: ${{ github.repository_owner == 'Red-Eyed' && github.actor == 'Red-Eyed' && needs.check.outputs.release_needed == 'true' }}
```

The expensive job runs only when the cheap check found an eligible ONNX Runtime version. If Maven
does not have the Android AARs yet, or if the actor is not `Red-Eyed`, the workflow stops here.

### Release Environment

```yaml
env:
  GH_TOKEN: ${{ github.token }}
  GITHUB_TOKEN: ${{ github.token }}
  ORT_RUNNER_ORT_VERSION: ${{ needs.check.outputs.ort_version }}
  ORT_RUNNER_CONTAINER_ENGINE: docker
```

`ORT_RUNNER_ORT_VERSION` tells `scripts/fetch_onnxruntime.py` which runtime to fetch for this
refresh build. The local default remains in the script, so developer builds do not need this env
var.

`ORT_RUNNER_CONTAINER_ENGINE=docker` switches CI to Docker. Local recipes still default to Podman,
but GitHub-hosted runners already have Docker and Docker integrates with the BuildKit cache used
below.

### Docker And Build Cache Setup

```yaml
- uses: docker/setup-qemu-action@v3
- uses: docker/setup-buildx-action@v3
```

The toolchain images are built as `linux/arm64`. QEMU makes that possible on GitHub's x64 hosted
runners. Buildx is used so Docker can restore and save image layers with GitHub's cache service.

```yaml
- uses: actions/cache@v4
  with:
    path: |
      .cargo
      sdk
      target
```

This cache is for repo-local build state:

- `.cargo`: crate sources used by `cargo --offline` inside the containers;
- `sdk`: downloaded ONNX Runtime SDK/AAR contents for the selected ORT version;
- `target`: compiled Rust outputs, useful when rerunning a failed release workflow.

### Host-Side Cargo Fetch

```yaml
- name: Fetch Rust dependencies
  run: |
    rustup toolchain install 1.97.0 --profile minimal
    CARGO_HOME="${GITHUB_WORKSPACE}/.cargo" cargo +1.97.0 fetch --locked
```

Container builds deliberately run Cargo with `--offline`. This step populates the bind-mounted
`.cargo` directory before the containers start, so a fresh GitHub runner does not fail cold.

### Toolchain Images

```yaml
- name: Build Linux Toolchain Image
  if: contains(needs.check.outputs.targets, 'linux-')
  uses: docker/build-push-action@v6
```

The Linux image is built only when at least one Linux target is selected by the check job.

```yaml
- name: Build Android Toolchain Image
  if: contains(needs.check.outputs.targets, 'android-')
  uses: docker/build-push-action@v6
```

The Android image is built only when Android targets are selected. In normal refresh releases this
is expected, because Android artifacts gate the workflow.

Both image steps use:

```yaml
cache-from: type=gha,scope=...
cache-to: type=gha,mode=max,scope=...
load: true
```

The `gha` cache stores expensive image layers, including Rust, system linkers and the Android NDK.
`load: true` makes the built image available to the later `docker run` calls in the same job.

### Target Build Loop

```yaml
- name: Build selected targets
  run: |
    for target in ${{ needs.check.outputs.targets }}; do
      uv run scripts/build.py "$target"
      uv run scripts/smoke.py "$target"
    done
```

For each selected target, `scripts/build.py`:

- fetches the selected ONNX Runtime SDK or AAR;
- runs the Rust release build inside the selected toolchain image;
- copies `libonnxruntime.so` beside the binary.

Then `scripts/smoke.py` verifies the binary architecture and confirms the runtime library was
bundled. Linux targets also run `ort_runner --version` when the build image can execute that
target.

### Publish

```yaml
- name: Publish release
  run: uv run scripts/release_onnxruntime.py ${{ needs.check.outputs.targets }}
```

`scripts/release_onnxruntime.py` packages the selected targets and creates a GitHub release named
`onnxruntime-vX.Y.Z`. The release notes record the `ort_runner` version, ONNX Runtime version,
upstream ONNX Runtime release URL and packaged targets.
