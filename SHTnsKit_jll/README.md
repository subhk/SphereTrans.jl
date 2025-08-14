SHTnsKit_jll (local, path-based)

This is a minimal Julia JLL-style wrapper that loads a prebuilt `libshtns_omp.so` from a local artifact path. It is ideal for local development and CI when you already have a working SHTns OpenMP build and do not want to go through BinaryBuilder/Yggdrasil yet.

Contents
- `Artifacts.toml` declares an artifact named `SHTnsKit` using a local `path` entry.
- `src/SHTnsKit_jll.jl` loads the library product `libshtns_omp` from `lib/libshtns_omp.so` within the artifact directory.
- `local_artifacts/current/lib/libshtns_omp.so` is where the shared library is looked up.

Usage
1) Put your library in place
   - Copy your `libshtns_omp.so` into `SHTnsKit_jll/local_artifacts/current/lib/` (already done here).

2) Develop this package in your Julia environment
   - In the SHTnsKit.jl environment run:
     ] dev ./SHTnsKit_jll

3) Load and use from Julia
   - using SHTnsKit_jll
   - SHTnsKit_jll.libshtns_omp  # Pass this as the library handle in your ccalls

Notes
- This package uses a path-based artifact. That is convenient for local use but not suitable for registration. For distribution, you would replace the `path` entry with platform-specific downloads or host the artifact content and reference it by `git-tree-sha1`.
- If you also need a non-OpenMP `libshtns.so`, mirror this pattern by adding another `@declare_library_product` and placing the appropriate file under `local_artifacts/current/lib/`.

Release (proper artifacts)
- See DEVELOPING.md for building platform tarballs with BinaryBuilder and converting `Artifacts.toml` from a path-based entry to platform-resolved downloads suitable for CI and registration.
