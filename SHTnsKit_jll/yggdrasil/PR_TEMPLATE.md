Title: SHTnsKit_jll: SHTns with OpenMP and FFTW

Summary
- Builds the SHTns library with OpenMP enabled and links to FFTW.
- Provides cross-platform binaries (Linux, macOS, Windows; x86_64 + aarch64 where applicable).

Checklist
- [ ] name = "SHTnsKit" and version set appropriately in build_tarballs.jl
- [ ] Sources pin stable SHTns commit/tag
- [ ] OpenMP and FFTW dependencies declared
- [ ] Products export libshtns (and optionally libshtns_omp)
- [ ] CI: build succeeds for listed platforms

Notes
- This package intentionally provides an alternative to SHTns_jll with specific build flags suitable for SHTnsKit.jl use.
- If maintainers prefer, this could be an enhancement to the existing SHTns_jll instead; build flags may be aligned accordingly.

