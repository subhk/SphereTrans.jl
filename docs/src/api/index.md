# API Reference

Complete reference for all SHTnsKit.jl functions and types.

## Configuration Management

### Configuration Creation

```julia
create_config(lmax; mmax=lmax, mres=1, grid_type=SHT_GAUSS, norm=SHT_ORTHONORMAL, T=Float64) → SHTnsConfig
```
Create a new SHTns configuration with specified parameters. After creation, call `set_grid!(cfg, nlat, nphi)`.

**Arguments:**
- `lmax::Int`: Maximum spherical harmonic degree
- `mmax::Int`: Maximum spherical harmonic order (default: lmax)
- `mres::Int`: Azimuthal resolution (default: 1)
- `grid_type::SHTnsGrid`: `SHT_GAUSS` or `SHT_REGULAR`
- `norm::SHTnsNorm`: Normalization convention
- `T::Type`: Floating point precision (default: Float64)

**Returns:** `SHTnsConfig` object

---

```julia
create_gauss_config(lmax, mmax) → SHTnsConfig
```
Create configuration with optimal Gauss-Legendre grid.

**Arguments:**
- `lmax::Int`: Maximum degree
- `mmax::Int`: Maximum order

**Returns:** `SHTnsConfig` with Gauss grid setup

**Example:**
```julia
cfg = create_gauss_config(32, 32)
nlat, nphi = get_nlat(cfg), get_nphi(cfg)  # 33 × 65
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
nlat, nphi = get_nlat(cfg), get_nphi(cfg)  # 65 × 65
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

For triangular truncation: `nlm = (lmax+1) × (lmax+2) / 2`

---

```julia
get_index(cfg::SHTnsConfig, l::Int, m::Int) → Int
```
Get linear index for spherical harmonic Y_l^m.

**Arguments:**
- `l::Int`: Degree (0 ≤ l ≤ lmax)
- `m::Int`: Order (-l ≤ m ≤ l)

**Returns:** Linear index in spectral coefficient array

### Grid Information

```julia
get_coordinates(cfg::SHTnsConfig) → (θ::Matrix, φ::Matrix)
```
Get colatitude and longitude coordinate matrices.

**Returns:**
- `θ::Matrix{Float64}`: Colatitude (0 to π)
- `φ::Matrix{Float64}`: Longitude (0 to 2π)

---

```julia
get_gauss_weights(cfg::SHTnsConfig) → Vector{Float64}
```
Get Gaussian quadrature weights (for Gauss grids only).

**Returns:** Vector of integration weights

### Configuration Cleanup

```julia
free_config(cfg::SHTnsConfig) → Nothing
```
Free memory associated with configuration. Always call after use.

**Example:**
```julia
cfg = create_gauss_config(32, 32)
# ... use configuration ...
free_config(cfg)
```

## Scalar Field Transforms

### Memory Allocation

```julia
allocate_spectral(cfg::SHTnsConfig) → Vector{Float64}
```
Allocate array for spectral coefficients.

**Returns:** Zero-initialized vector of length `get_nlm(cfg)`

---

```julia
allocate_spatial(cfg::SHTnsConfig) → Matrix{Float64}
```
Allocate array for spatial field values.

**Returns:** Zero-initialized matrix of size `(get_nlat(cfg), get_nphi(cfg))`

### Forward Transform (Synthesis)

```julia
synthesize(cfg::SHTnsConfig, sh::Vector) → Matrix{Float64}
```
Transform from spectral to spatial domain (spherical harmonic synthesis).

**Arguments:**
- `sh::Vector{Float64}`: Spectral coefficients of length `get_nlm(cfg)`

**Returns:** Spatial field matrix `(nlat × nphi)`

**Example:**
```julia
cfg = create_gauss_config(16, 16)
sh = rand(get_nlm(cfg))
spatial = synthesize(cfg, sh)  # 17×33 matrix
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

**Returns:** Spectral coefficients vector of length `get_nlm(cfg)`

**Example:**
```julia
cfg = create_gauss_config(16, 16)
spatial = rand(get_nlat(cfg), get_nphi(cfg))
sh = analyze(cfg, spatial)
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
sh_complex = rand(ComplexF64, get_nlm(cfg))
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
S_lm = rand(get_nlm(cfg))  # Spheroidal
T_lm = rand(get_nlm(cfg))  # Toroidal
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

### Gradient and Curl Operations

```julia
compute_gradient(cfg::SHTnsConfig, scalar_lm::Vector) → (∇θ::Matrix, ∇φ::Matrix)
```
Compute gradient of scalar field (produces spheroidal vector field).

**Arguments:**
- `scalar_lm::Vector{Float64}`: Scalar field spectral coefficients

**Returns:**
- `∇θ::Matrix{Float64}`: ∂/∂θ component  
- `∇φ::Matrix{Float64}`: (1/sin θ)∂/∂φ component

---

```julia
compute_curl(cfg::SHTnsConfig, toroidal_lm::Vector) → (curlθ::Matrix, curlφ::Matrix)
```
Compute curl of toroidal field.

**Arguments:**
- `toroidal_lm::Vector{Float64}`: Toroidal field coefficients

**Returns:**  
- `curlθ::Matrix{Float64}`: Curl θ component
- `curlφ::Matrix{Float64}`: Curl φ component

## Field Rotations

### Spectral Domain Rotation

```julia
rotate_field(cfg::SHTnsConfig, sh::Vector, α::Real, β::Real, γ::Real) → Vector{Float64}
```
Rotate field using Wigner D-matrices in spectral domain.

**Arguments:**
- `sh::Vector{Float64}`: Input spectral coefficients
- `α, β, γ::Real`: Euler angles (ZYZ convention)

**Returns:** Rotated spectral coefficients

**Example:**
```julia
cfg = create_gauss_config(16, 16)  
sh = rand(get_nlm(cfg))
# Rotate by 45° around Z, 60° around Y, 30° around Z
sh_rotated = rotate_field(cfg, sh, π/4, π/3, π/6)
```

### Spatial Domain Rotation

```julia
rotate_spatial_field(cfg::SHTnsConfig, spatial::Matrix, α::Real, β::Real, γ::Real) → Matrix{Float64}
```
Rotate spatial field (analysis → rotate → synthesis).

**Arguments:**
- `spatial::Matrix{Float64}`: Input spatial field
- `α, β, γ::Real`: Euler angles

**Returns:** Rotated spatial field

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
sh = rand(get_nlm(cfg))
power = power_spectrum(cfg, sh)  # Length lmax+1
# power[1] = l=0 power, power[2] = l=1 power, etc.
```

## Threading Control

### Thread Management

```julia
get_num_threads() → Int
```
Get current number of OpenMP threads.

---

```julia
set_num_threads(nthreads::Int) → Nothing
```
Set number of OpenMP threads for SHTns operations.

**Arguments:**
- `nthreads::Int`: Number of threads (≥ 1)

---

```julia
set_optimal_threads() → Nothing  
```
Set thread count to optimal value for current system.

**Example:**
```julia
println("Default threads: ", get_num_threads())
set_num_threads(4)
println("Set to 4 threads: ", get_num_threads())
set_optimal_threads()
println("Optimal threads: ", get_num_threads())
```

## GPU Support (CUDA Extension)

GPU functions are available when CUDA.jl is loaded and functional.

### GPU Memory Management

```julia
initialize_gpu(device_id::Int=0; verbose::Bool=true) → Bool
```
Initialize GPU for SHTns operations.

**Arguments:**
- `device_id::Int`: GPU device ID
- `verbose::Bool`: Print initialization messages

**Returns:** `true` if successful

---

```julia
cleanup_gpu(; verbose::Bool=true) → Bool
```
Clean up GPU resources.

### GPU Transforms

```julia
synthesize_gpu(cfg::SHTnsConfig, sh_gpu::CuArray) → CuArray
```
GPU-accelerated synthesis.

**Arguments:**
- `sh_gpu::CuArray{Float64}`: Spectral coefficients on GPU

**Returns:** Spatial field on GPU

---

```julia
analyze_gpu(cfg::SHTnsConfig, spatial_gpu::CuArray) → CuArray
```
GPU-accelerated analysis.

**Example:**
```julia
using CUDA
if CUDA.functional()
    cfg = create_gpu_config(32, 32)
    sh = rand(get_nlm(cfg))
    sh_gpu = CuArray(sh)
    
    spatial_gpu = synthesize_gpu(cfg, sh_gpu)
    spatial_cpu = Array(spatial_gpu)  # Copy back to CPU
    
    free_config(cfg)
end
```

## MPI Support (MPI Extension)

MPI functions are available when MPI.jl is loaded.

### MPI Configuration

```julia
create_mpi_config(lmax::Int, mmax::Int, comm=MPI.COMM_WORLD) → SHTnsConfig
```
Create MPI-distributed configuration.

**Arguments:**
- `lmax, mmax::Int`: Maximum degree and order
- `comm`: MPI communicator

**Returns:** MPI-enabled configuration

### Distributed Transforms

```julia
mpi_synthesize(cfg::SHTnsConfig, sh_local::Vector) → Matrix{Float64}
```
MPI-distributed synthesis.

---

```julia
mpi_analyze(cfg::SHTnsConfig, spatial_local::Matrix) → Vector{Float64}
```
MPI-distributed analysis.

## Constants and Flags

### SHTnsFlags Module

```julia
SHTnsFlags.SHT_GAUSS           # Gauss-Legendre grid
SHTnsFlags.SHT_REGULAR_GRID    # Regular equiangular grid  
SHTnsFlags.SHT_QUICK_INIT      # Fast initialization
SHTnsFlags.SHT_REAL_NORM       # Real normalization
SHTnsFlags.SHT_NO_CS_PHASE     # No Condon-Shortley phase
SHTnsFlags.SHT_FOURPI          # 4π normalization
SHTnsFlags.SHT_SCHMIDT         # Schmidt normalization
```

**Example:**
```julia
# Create configuration with specific flags
flags = SHTnsFlags.SHT_GAUSS | SHTnsFlags.SHT_REAL_NORM
cfg = create_config(32, 32, 65, flags)
```

## Error Handling

### Common Errors

- **`BoundsError`**: Invalid lmax/mmax values
- **`AssertionError`**: Array size mismatches
- **`LoadError`**: SHTns library not found

### Best Practices

```julia
# Always check array sizes
@assert length(sh) == get_nlm(cfg) "Wrong spectral array size"
@assert size(spatial) == (get_nlat(cfg), get_nphi(cfg)) "Wrong spatial array size"

# Always free configurations
try
    cfg = create_gauss_config(32, 32)
    # ... work with cfg ...
finally
    free_config(cfg)
end

# Or use do-block pattern (if implemented)
with_config(create_gauss_config(32, 32)) do cfg
    # ... work with cfg ...
    # automatically freed
end
```

## Automatic Differentiation Support

### Helper Functions

```julia
get_lm_from_index(cfg::SHTnsConfig, idx::Int) → (l::Int, m::Int)
```
Get spherical harmonic degree and order from linear index.

**Arguments:**
- `idx::Int`: Linear index (1-based) in spectral coefficient array

**Returns:**
- `l::Int`: Spherical harmonic degree (0 ≤ l ≤ lmax)
- `m::Int`: Spherical harmonic order (-l ≤ m ≤ l)

**Example:**
```julia
cfg = create_gauss_config(8, 8)
l, m = get_lm_from_index(cfg, 1)  # Returns (0, 0) for Y₀⁰
l, m = get_lm_from_index(cfg, 4)  # Returns (1, 1) for Y₁¹
```

---

```julia
get_index_from_lm(cfg::SHTnsConfig, l::Int, m::Int) → Int
```
Get linear index from spherical harmonic degree and order.

**Arguments:**
- `l::Int`: Spherical harmonic degree
- `m::Int`: Spherical harmonic order

**Returns:**
- `idx::Int`: Linear index in spectral coefficient array

**Example:**
```julia
cfg = create_gauss_config(8, 8)
idx = get_index_from_lm(cfg, 0, 0)  # Returns 1 for Y₀⁰
idx = get_index_from_lm(cfg, 1, 1)  # Returns 4 for Y₁¹
```

### ForwardDiff.jl Integration

When ForwardDiff.jl is available, all SHTnsKit transform functions automatically support forward-mode automatic differentiation:

```julia
using SHTnsKit, ForwardDiff

cfg = create_gauss_config(16, 16)

function objective(sh_params)
    spatial = synthesize(cfg, sh_params)
    return sum(spatial.^2)
end

sh = rand(get_nlm(cfg))
gradient = ForwardDiff.gradient(objective, sh)
```

**Supported Functions:**
- `synthesize`, `synthesize!`
- `analyze`, `analyze!`  
- `synthesize_complex`, `analyze_complex`
- `synthesize_vector`, `analyze_vector`
- `compute_gradient`, `compute_curl`
- `power_spectrum`

### Zygote.jl Integration

When Zygote.jl is available, all SHTnsKit transform functions automatically support reverse-mode automatic differentiation:

```julia
using SHTnsKit, Zygote

cfg = create_gauss_config(16, 16)

function loss(spatial_field)
    sh = analyze(cfg, spatial_field)  
    return sum(sh[1:10].^2)  # Focus on low modes
end

spatial = rand(get_nlat(cfg), get_nphi(cfg))
gradient = Zygote.gradient(loss, spatial)[1]
```

**Key Features:**
- Leverages linearity of spherical harmonic transforms
- Efficient adjoint computations using transform duality
- Full support for complex and vector field operations
- Memory-efficient implementation

### Performance Notes

**Forward vs Reverse Mode Selection:**
- Use **ForwardDiff** when: `n_parameters < n_outputs`
- Use **Zygote** when: `n_parameters > n_outputs`

**Memory Optimization:**
- Pre-allocate buffers for repeated AD computations
- Use in-place operations (`synthesize!`, `analyze!`) when possible
- Consider chunking for very large problems
