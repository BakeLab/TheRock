# BakeLab SDK publication

Gitea's `publish-sdk` workflow runs for `trim` pushes or manual dispatch. It builds
only `gfx1201` / `znver4`, using the shared Breadkernel build and publish scripts.
The triggering TheRock commit is pinned, checked against the SDK artifact manifest,
and included in the published version. The upstream multi-architecture workflows
are not part of this release path.

Both jobs reuse the normal CPU/Linux/x64 runner labels. The SDK job needs a
CPU-only Arch Linux guest with a znver4-compatible CPU, 64 GiB RAM, a 500 GiB
disk and a 24-hour task limit. Apply these resource requirements only to this
SDK build through trusted Gitea configuration, not to every CPU runner. It must not attach the old
ROCm accelerator runtime. The current build sets `-march=znver4` for both SDK
libraries and build tools that execute during compilation, so a generic CPU
profile is insufficient. The workflow checks CPU features, Python 3.14+, free
space and memory before compiling. GPU access is not required. The current accelerator capability overlays do not
configure CPU resources or task lifetime; deployment must supply those limits
before dispatch. Removing a label alone does not provision build capacity.

Publishing reuses Gitea Actions OIDC and `uv publish` through Breadkernel.
Configure cross-repository read access to Breadkernel and Breadboard, and exact
OIDC grants for this repository, `refs/heads/trim` and
`.gitea/workflows/publish-sdk.yml`:

- `bb-dev-rocm-znver4`: `rocm`, `rocm-sdk-core`, `rocm-sdk-libraries`,
  `rocm-sdk-device-gfx1201`, `rocm-sdk-devel`, and optional `rocm-profiler`.
- `BakeLab`: `breadboard-guest-image`, `breadboard-board-pack`.
- `bb-dev-rocm-znver4`: `breadboard-accel-pack`, `breadboard-asset-lock`.

After SDK publication succeeds, a separate CPU guest builds the ROCm runtime
pack with that exact SDK version and publishes its compatibility lock last.
This avoids requiring an older ROCm pack to build its replacement. Node updates
consume the published lock through the existing updater; the workflow does not
change host drivers or restart production services. Compiling and publishing
packages does not establish a successful GPU workload.

Deploy Breadkernel's frozen `THEROCK_REF` support and the `rocm-version` action
input before enabling this workflow.
