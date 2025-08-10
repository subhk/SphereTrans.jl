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
