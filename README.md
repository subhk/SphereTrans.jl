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
GPU-friendly wrappers (`analyze_gpu`/`synthesize_gpu`) stage through host but
benefit from the native GPU acceleration internally.
