## Grid Types

SHTnsKit supports several latitude grids selected via `shtns_set_grid` flags:

- Gauss (Gaussian quadrature): Exact for integrals up to degree `2*nlat-1`.
  Use for highest accuracy. Suggested `nlat = lmax+1`, `nphi ≥ 2*mmax+1`.

- Regular equiangular without poles (reg_fast/reg_dct/quick_init):
  Midpoint latitudes `θ_i = (i+0.5)π/nlat`. Fast to set up and compatible with
  FFT-friendly sampling. Weights `w_i = (π/nlat) sin(θ_i)` approximate the integral.

- Regular equiangular with poles (reg_poles):
  `θ_i = i π/(nlat-1)` including poles. Weights are `w_i = (π/(nlat-1)) sin(θ_i)`.

Use `shtns_set_grid_auto` to get suggested `nlat`/`nphi` depending on grid type.
For best numerical exactness, prefer Gauss. For speed and compatibility with
standard image-like sampling, prefer regular grids with precomputed Legendre
tables via `prepare_plm_tables!(cfg)`.

