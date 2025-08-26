# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [1.1.0] - 2025-08-26

Highlights
- Full MPI mirror using PencilArrays/PencilFFTs for scalar, vector (sphtor) and QST transforms
- New distributed rotations (Z, Y with truncated/allgather, Euler Z–Y–Z, 90° shortcuts)
- Distributed AD convenience wrappers (Zygote/ForwardDiff) for PencilArrays
- Distributed local/point evaluations and packed LM helpers
- New Distributed Guide for users and contributors

Added
- Distributed scalar transforms: `dist_analysis`, `dist_synthesis` with optional rFFT/irFFT
- Distributed vector (sphtor) and QST transforms with PLM tables, Robert form, normalization/phase support
- Plans carrying `use_rfft`: `DistAnalysisPlan`, `DistPlan`, `DistSphtorPlan`, `DistQstPlan`
- Distributed diagnostics on PencilArrays: energy/enstrophy spectra and grid energies
- Rotations on PencilArrays: `dist_SH_Zrotate` (in/out), `dist_SH_Yrotate` (trunc-gather), `dist_SH_Yrotate_allgatherm!`, `dist_SH_rotate_euler`, `dist_SH_Yrotate90`, `dist_SH_Xrotate90`
- Packed-vector rotation wrappers (dense Qlm → distributed → dense Rlm): `dist_SH_*_packed`
- Local/point evaluations: `dist_SH_to_point`, `dist_SH_to_lat`, `dist_SHqst_to_point`, `dist_SHqst_to_lat`
- Packed conversions (dense/LM ⇄ distributed): real and complex (LM_cplx)
- AD wrappers for PencilArrays: `zgrad_scalar_energy`, `zgrad_vector_energy`, `fdgrad_scalar_energy`, `fdgrad_vector_energy`
- Comprehensive guide: `docs/Distributed_SHTnsKit_Guide.md` and Documenter page `docs/src/distributed.md`

Changed
- Default distributed Y-rotation wrapper `dist_SH_Yrotate` now uses truncated gather per l (bandwidth reduction)
- README updated with Distributed API cheatsheet and guide link

Removed
- Old docs workflow `.github/workflows/documentation.yml` (docs build handled in `ci.yml`)

Notes
- Enable FFT plan caching by setting `ENV["SHTNSKIT_CACHE_PENCILFFTS"] = "1"`
- Prefer truncated-gather Y-rotation for broad spectra; use `dist_SH_Yrotate_allgatherm!` if most energy is at high m

## [1.0.0] - 2025-08-10
- Initial release
