# SHTnsKit.jl

Tools for Spherical Harmonics transformations based on the SHTns C library.

## Wrappers

The package currently exposes a minimal set of low-level wrappers around the
SHTns C API:

- `create_config` – construct a configuration via `shtns_create_with_opts`.
- `set_grid` – set the spatial grid used for transforms.
- `sh_to_spat` and `spat_to_sh` – perform synthesis and analysis transforms.
- `get_lmax` / `get_mmax` – query maximal degree and order of a configuration.
- `get_nlat` / `get_nphi` – obtain grid dimensions.
- `get_nlm` – number of spectral coefficients.
- `lmidx` – index helper for `(l,m)` pairs.
- `free_config` – release resources allocated for a configuration.

These wrappers expect the `libshtns.so` shared library to be available on the
system. They provide the building blocks for higher level Julia interfaces.

## High-level API

On top of the low-level calls, the package provides shape-safe, allocating
wrappers and GPU-friendly helpers:

- `allocate_spectral(cfg)` / `allocate_spatial(cfg)`: convenience allocation for
  spectral vectors and spatial grids.
- `synthesize!(cfg, sh, spat)` / `synthesize(cfg, sh)`: spectral → spatial
  transforms with `spat` treated as a `(nlat, nphi)` matrix.
- `analyze!(cfg, spat, sh)` / `analyze(cfg, spat)`: spatial → spectral
  transforms with `spat` as `(nlat, nphi)` and `sh` a vector of length `nlm`.

All high-level transforms operate on `Float64` data, promoting inputs as needed.

### Threading

- High-level transforms are safe under Julia multi-threading: calls using the
  same `SHTnsConfig` are serialized via an internal lock to avoid races.
- Different configurations may run concurrently in parallel threads.
- SHTns itself may use internal OpenMP threads; to avoid oversubscription, set
  either Julia threads or SHTns OpenMP threads conservatively.

### GPU/CUDA usage

If you work with CUDA arrays (e.g., `CUDA.CuArray`), you can use:

- `synthesize_gpu(cfg, sh_dev)` → returns a device matrix `(nlat, nphi)`
- `analyze_gpu(cfg, spat_dev)` → returns a device vector of length `nlm`

These functions stage data to host memory, call the CPU SHTns routines, and copy
results back to the device. They do not require importing CUDA in this package;
they operate generically on any device array type that supports `Array(x)`,
`similar(x, T, dims...)`, and `copyto!` (which `CUDA.CuArray` does).

Example:

```julia
using SHTnsKit
# optionally: using CUDA

cfg = create_config(16, 16, 1)
set_grid(cfg, 64, 128, 0)

# CPU
sh = allocate_spectral(cfg)
rand!(sh)
spat = synthesize(cfg, sh)          # (64, 128) matrix
sh2  = analyze(cfg, spat)           # length matches get_nlm(cfg)

# GPU (if CUDA is available)
# shd  = CUDA.CuArray(sh)
# spatd = synthesize_gpu(cfg, shd)   # device matrix
# shd2  = analyze_gpu(cfg, spatd)    # device vector

free_config(cfg)
```

Device-pointer fast path
- If your SHTns GPU entrypoints expect device pointers and run kernels
  internally, opt-in to a zero-copy path that avoids host staging.
  - Set env: `SHTNSKIT_GPU_PTRKIND=device`
  - Or call: `SHTnsKit.SHTnsKitCUDAExt.enable_gpu_deviceptrs!()`
- Combine this with native GPU entrypoints (see below) for best performance.

### MPI (distributed SHT)

This package exposes a lightweight MPI extension that can call SHTns's native
MPI entrypoints when available. Symbol names vary across builds, so the
extension dynamically loads them via environment variables or an explicit call:

- `SHTNSKIT_MPI_CREATE`   – MPI config constructor symbol
- `SHTNSKIT_MPI_SET_GRID` – grid setup symbol
- `SHTNSKIT_MPI_SH2SPAT`  – spectral→spatial symbol
- `SHTNSKIT_MPI_SPAT2SH`  – spatial→spectral symbol
- `SHTNSKIT_MPI_FREE`     – destructor symbol

When present, you can do per-rank distributed transforms using:

```julia
using MPI, SHTnsKit
MPI.Init()
comm = MPI.COMM_WORLD

# Enable native SHTns MPI entrypoints by name
SHTnsKit.SHTnsKitMPIExt.enable_native_mpi!(; create="...", set_grid="...",
    sh2spat="...", spat2sh="...", free="...")

cfg = SHTnsKit.SHTnsKitMPIExt.create_mpi_config(comm, 16, 16, 1)
set_grid(cfg, 64, 128, 0)
sh = allocate_spectral(cfg.cfg)          # allocation uses cfg.cfg underneath
spat = allocate_spatial(cfg.cfg)
synthesize!(cfg, sh, spat)
analyze!(cfg, spat, sh)
free_config(cfg)
MPI.Finalize()
```

Notes:
- If you don’t enable MPI entrypoints, the extension transparently falls back
  to per-rank CPU SHTns (each rank creates its own `SHTnsConfig`).
- Precise C signatures may differ across builds; if your entrypoints require
  explicit `MPI_Comm` arguments or other parameters, please share the exact C
  prototypes and we will wire exact wrappers.

### Native SHTns GPU entrypoints (optional)

If your `libshtns` build provides GPU-accelerated entrypoints that are drop-in
replacements for the CPU functions (same C signatures), you can opt-in at
runtime by setting environment variables before loading the package:

- `SHTNSKIT_GPU_SH2SPAT`: symbol name for spectral→spatial function
- `SHTNSKIT_GPU_SPAT2SH`: symbol name for spatial→spectral function

Example (Unix shells):

```bash
export SHTNSKIT_GPU_SH2SPAT=shtns_sh_to_spat_gpu
export SHTNSKIT_GPU_SPAT2SH=shtns_spat_to_sh_gpu
julia -e 'using SHTnsKit; @show SHTnsKit.is_native_gpu_enabled()'
```

Or programmatically:

```julia
using SHTnsKit
SHTnsKit.enable_native_gpu!(; sh2spat="shtns_sh_to_spat_gpu",
                                spat2sh="shtns_spat_to_sh_gpu")
```

When enabled, high-level `analyze!`/`synthesize!` use these entrypoints; the
GPU-friendly wrappers (`analyze_gpu`/`synthesize_gpu`) will either stage through
host memory (default) or pass device pointers directly if
`SHTNSKIT_GPU_PTRKIND=device` is set or `enable_gpu_deviceptrs!()` was called.

### Default Symbols

If environment variables are not set, SHTnsKit tries common default symbol
names for dynamic resolution:

- CPU vector transforms:
  - torpol2uv: `shtns_torpol2uv`
  - uv2torpol: `shtns_uv2torpol`
- GPU scalar transforms:
  - sh_to_spat: `shtns_sh_to_spat_gpu`
  - spat_to_sh: `shtns_spat_to_sh_gpu`
- MPI scalar transforms:
  - create: `shtns_mpi_create_with_opts`
  - set_grid: `shtns_mpi_set_grid`
  - sh_to_spat: `shtns_mpi_sh_to_spat`
  - spat_to_sh: `shtns_mpi_spat_to_sh`
  - free: `shtns_mpi_free`
- MPI vector transforms:
  - torpol2uv: `shtns_mpi_torpol2uv`
  - uv2torpol: `shtns_mpi_uv2torpol`
- Grid latitudes (optional):
  - get_theta: `shtns_get_theta`
