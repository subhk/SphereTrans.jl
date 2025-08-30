# API Reference

Complete reference for all SHTnsKit.jl functions and types.

## Configuration Management

### Configuration Creation

```julia
create_config(lmax; mmax=lmax, mres=1, nlat=lmax+2, nlon=max(2*lmax+1,4),
              norm=:orthonormal, cs_phase=true, real_norm=false, robert_form=false,
              grid_type=:gauss) → SHTConfig
```
Create a configuration with specified parameters. Currently `grid_type = :gauss` is supported and forwards to `create_gauss_config`.

Notes:
- Auto-corrects undersized grids to satisfy accuracy constraints:
  - Ensures `nlat ≥ lmax+1` (Gauss–Legendre quadrature exactness)
  - Ensures `nlon ≥ 2*mmax+1` (resolve azimuthal orders up to `mmax`)
- A legacy form `create_config(::Type{T}, lmax, nlat, mres; ...)` is also accepted; the type is ignored.

**Example:**
```julia
cfg = create_config(32; nlat=30, nlon=60)  # adjusted to nlat=33, nlon=65
```

---

```julia
create_gauss_config(lmax, nlat; mmax=lmax, mres=1, nlon=max(2*lmax+1,4),
                    norm=:orthonormal, cs_phase=true, real_norm=false,
                    robert_form=false) → SHTConfig
```
Create configuration with Gauss–Legendre grid. Requires `nlat ≥ lmax+1` and `nlon ≥ 2*mmax+1`.

**Example:**
```julia
cfg = create_gauss_config(32, 34; nlon=65)
nlat, nphi = cfg.nlat, cfg.nlon  # 34 × 65
```

---

```julia
create_regular_config(lmax, mmax) → SHTnsConfig
```
Create configuration with regular equiangular grid.

**Arguments:**
- `lmax::Int`: Maximum degree  
- `mmax::Int`: Maximum order

**Returns:** `SHTnsConfig` with regular grid setup

**Example:**
```julia
cfg = create_regular_config(32, 32)
nlat, nphi = cfg.nlat, cfg.nlon  # 65 × 65
```

---

<!-- GPU configuration is not supported in this package -->

### Configuration Queries

```julia
get_lmax(cfg::SHTnsConfig) → Int
```
Get maximum spherical harmonic degree.

---

```julia
get_mmax(cfg::SHTnsConfig) → Int  
```
Get maximum spherical harmonic order.

---

```julia
get_nlat(cfg::SHTnsConfig) → Int
```
Get number of latitude points in spatial grid.

---

```julia
get_nphi(cfg::SHTnsConfig) → Int
```
Get number of longitude points in spatial grid.

---

```julia
get_nlm(cfg::SHTnsConfig) → Int
```
Get total number of (l,m) spectral coefficients.

For the real basis, coefficients are stored for m ≥ 0.

---

```julia
lmidx(cfg::SHTnsConfig, l::Int, m::Int) → Int
```
Get linear index (1‑based) for spherical harmonic coefficient (l, m≥0).

### Grid Information

```julia
get_theta(cfg::SHTnsConfig, i::Int) → Real
get_phi(cfg::SHTnsConfig, j::Int) → Real
```
Access single grid coordinates by index.

```julia
SHTnsKit.create_coordinate_matrices(cfg::SHTnsConfig) → (θ::Matrix, φ::Matrix)
```
Create colatitude and longitude matrices for the grid.

---

```julia
get_gauss_weights(cfg::SHTnsConfig) → Vector{Float64}
```
Get Gaussian quadrature weights (for Gauss grids only).

**Returns:** Vector of integration weights

### Configuration Cleanup

```julia
destroy_config(cfg::SHTnsConfig) → Nothing
```
Free memory associated with configuration. Always call after use.

**Example:**
```julia
cfg = create_gauss_config(32, 32)
# ... use configuration ...
destroy_config(cfg)
```

## Scalar Field Transforms

### Memory Allocation

```julia
allocate_spectral(cfg::SHTnsConfig) → Vector{Float64}
```
Allocate array for spectral coefficients.

**Returns:** Zero-initialized vector of length `cfg.nlm`

---

```julia
allocate_spatial(cfg::SHTnsConfig) → Matrix{Float64}
```
Allocate array for spatial field values.

**Returns:** Zero-initialized matrix of size `(cfg.nlat, cfg.nlon)`

### Forward Transform (Synthesis)

```julia
synthesize(cfg::SHTnsConfig, sh::Vector) → Matrix{Float64}
```
Transform from spectral to spatial domain (spherical harmonic synthesis).

**Arguments:**
- `sh::Vector{Float64}`: Spectral coefficients of length `cfg.nlm`

**Returns:** Spatial field matrix `(nlat × nphi)`

**Example:**
```julia
cfg = create_gauss_config(16, 16)
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
spatial = synthesis(cfg, sh)  # 17×33 matrix
```

---

```julia
synthesize!(cfg::SHTnsConfig, sh::Vector, spatial::Matrix) → Nothing
```
In-place synthesis (avoids allocation).

**Arguments:**
- `sh::Vector{Float64}`: Input spectral coefficients
- `spatial::Matrix{Float64}`: Output spatial field (modified)

### Backward Transform (Analysis)

```julia
analyze(cfg::SHTnsConfig, spatial::Matrix) → Vector{Float64}
```
Transform from spatial to spectral domain (spherical harmonic analysis).

**Arguments:**
- `spatial::Matrix{Float64}`: Spatial field `(nlat × nphi)`

**Returns:** Spectral coefficients vector of length `cfg.nlm`

**Example:**
```julia
cfg = create_gauss_config(16, 16)
# Create bandlimited spatial field (smooth test function)
θ, φ = cfg.θ, cfg.φ
spatial = zeros(cfg.nlat, cfg.nlon)
for i in 1:cfg.nlat, j in 1:cfg.nlon
    spatial[i,j] = 1.0 + 0.5 * cos(θ[i]) + 0.3 * sin(θ[i]) * cos(φ[j])
end
sh = analysis(cfg, spatial)
```

---

```julia
analyze!(cfg::SHTnsConfig, spatial::Matrix, sh::Vector) → Nothing
```
In-place analysis (avoids allocation).

**Arguments:**
- `spatial::Matrix{Float64}`: Input spatial field
- `sh::Vector{Float64}`: Output spectral coefficients (modified)

## Complex Field Transforms

### Memory Allocation

```julia
allocate_complex_spectral(cfg::SHTnsConfig) → Vector{ComplexF64}
```
Allocate array for complex spectral coefficients.

---

```julia
allocate_complex_spatial(cfg::SHTnsConfig) → Matrix{ComplexF64}
```
Allocate array for complex spatial field values.

### Complex Transforms

```julia
synthesize_complex(cfg::SHTnsConfig, sh::Vector{ComplexF64}) → Matrix{ComplexF64}
```
Complex field synthesis.

**Example:**
```julia
cfg = create_gauss_config(16, 16)
sh_complex = rand(ComplexF64, cfg.nlm)
spatial_complex = synthesize_complex(cfg, sh_complex)
```

---

```julia
analyze_complex(cfg::SHTnsConfig, spatial::Matrix{ComplexF64}) → Vector{ComplexF64}
```
Complex field analysis.

## Vector Field Transforms  

Vector fields on the sphere are decomposed into **spheroidal** and **toroidal** components:
- **Spheroidal**: Poloidal component (has radial component)
- **Toroidal**: Azimuthal component (purely horizontal)

### Vector Synthesis

```julia
synthesize_vector(cfg::SHTnsConfig, S_lm::Vector, T_lm::Vector) → (Vθ::Matrix, Vφ::Matrix)
```
Synthesize vector field from spheroidal and toroidal coefficients.

**Arguments:**
- `S_lm::Vector{Float64}`: Spheroidal (poloidal) coefficients
- `T_lm::Vector{Float64}`: Toroidal coefficients

**Returns:**
- `Vθ::Matrix{Float64}`: Colatitude component
- `Vφ::Matrix{Float64}`: Longitude component

**Example:**
```julia
cfg = create_gauss_config(20, 20)
# Create bandlimited vector field coefficients
S_lm = zeros(cfg.nlm)  # Spheroidal
T_lm = zeros(cfg.nlm)  # Toroidal
S_lm[1] = 1.0  # Basic spheroidal mode
if cfg.nlm > 3
    T_lm[3] = 0.5  # Basic toroidal mode
end
Vθ, Vφ = synthesize_vector(cfg, S_lm, T_lm)
```

### Vector Analysis

```julia
analyze_vector(cfg::SHTnsConfig, Vθ::Matrix, Vφ::Matrix) → (S_lm::Vector, T_lm::Vector)
```
Analyze vector field into spheroidal and toroidal components.

**Arguments:**
- `Vθ::Matrix{Float64}`: Colatitude component
- `Vφ::Matrix{Float64}`: Longitude component

**Returns:**
- `S_lm::Vector{Float64}`: Spheroidal coefficients
- `T_lm::Vector{Float64}`: Toroidal coefficients

### Spatial Operators

```julia
SHTnsKit.spatial_derivative_phi(cfg::SHTnsConfig, spatial::Matrix) → Matrix
```
Exact φ‑derivative using FFT in longitude.

```julia
SHTnsKit.spatial_divergence(cfg::SHTnsConfig, Vθ::Matrix, Vφ::Matrix) → Matrix
SHTnsKit.spatial_vorticity(cfg::SHTnsConfig, Vθ::Matrix, Vφ::Matrix) → Matrix
```
Divergence and vertical vorticity of tangential vector fields on the unit sphere.

## Field Rotations

```julia
rotate_real!(cfg::SHTnsConfig, real_coeffs; alpha=0.0, beta=0.0, gamma=0.0) → Vector
rotate_complex!(cfg::SHTnsConfig, cplx_coeffs; alpha=0.0, beta=0.0, gamma=0.0) → Vector{Complex}
```
Rotate spectral coefficients in‑place using ZYZ Euler angles. For real fields, use `rotate_real!` or convert with `real_to_complex_coeffs`/`complex_to_real_coeffs`.

## Power Spectrum Analysis

```julia
power_spectrum(cfg::SHTnsConfig, sh::Vector) → Vector{Float64}
```
Compute spherical harmonic power spectrum.

**Arguments:**
- `sh::Vector{Float64}`: Spectral coefficients

**Returns:** Power per degree `P(l) = Σₘ |aₗᵐ|²`

**Example:**
```julia
cfg = create_gauss_config(32, 32)
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
power = power_spectrum(cfg, sh)  # Length lmax+1
# power[1] = l=0 power, power[2] = l=1 power, etc.
```

## Threading Control

### Thread Management

```julia
set_threading!(flag::Bool) → Bool
get_threading() → Bool
set_fft_threads(n::Integer) → Int
get_fft_threads() → Int
set_optimal_threads!() → (threads::Int, fft_threads::Int)
```
Enable/disable package parallel loops, control FFTW threads, or set a sensible configuration.

**Example:**
```julia
summary = set_optimal_threads!()
println(summary)
set_fft_threads(4)
println(get_fft_threads())
```

<!-- GPU and MPI extensions are not applicable in this pure-Julia implementation. -->

## Error Handling

### Common Errors

- **`BoundsError`**: Invalid lmax/mmax values
- **`AssertionError` / `DimensionMismatch`**: Array size mismatches

### Best Practices

```julia
# Always check array sizes
@assert length(sh) == cfg.nlm "Wrong spectral array size"
@assert size(spatial) == (cfg.nlat, cfg.nlon) "Wrong spatial array size"

# Always destroy configurations
cfg = create_gauss_config(32, 32)
# ... work with cfg ...
destroy_config(cfg)
```

## Helper Functions

lm_from_index(cfg::SHTnsConfig, idx::Int) → (l::Int, m::Int)

---

lmidx(cfg::SHTnsConfig, l::Int, m::Int) → Int

<!-- Automatic differentiation specifics are omitted; functions are pure Julia and generally AD-friendly. -->
