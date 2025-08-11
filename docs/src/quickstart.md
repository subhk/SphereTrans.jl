# Quick Start Guide

This tutorial will get you up and running with SHTnsKit.jl for spherical harmonic transforms in just a few minutes.

## Your First Transform

```julia
using SHTnsKit

# Create a spherical harmonic configuration
lmax = 32        # Maximum degree
mmax = 32        # Maximum order (typically same as lmax)
cfg = create_gauss_config(lmax, mmax)

# Generate some random spectral coefficients
sh_coeffs = rand(get_nlm(cfg))

# Transform from spectral to spatial domain (synthesis)
spatial_field = synthesize(cfg, sh_coeffs)

# Transform back to spectral domain (analysis)
recovered_coeffs = analyze(cfg, spatial_field)

# Check accuracy (should be very small)
error = norm(sh_coeffs - recovered_coeffs)
println("Round-trip error: $error")

# Clean up
free_config(cfg)
```

## Understanding the Basics

### Spectral vs Spatial Domains

- **Spectral Domain**: Coefficients of spherical harmonics Y_l^m(θ,φ)
- **Spatial Domain**: Values on a grid of points on the sphere

```julia
cfg = create_gauss_config(16, 16)

# Spectral: array of spherical harmonic coefficients
nlm = get_nlm(cfg)        # Number of (l,m) coefficients  
sh = zeros(nlm)           # Spectral coefficients
sh[1] = 1.0               # Set Y_0^0 = constant field

# Spatial: 2D array of values on sphere
nlat, nphi = get_nlat(cfg), get_nphi(cfg)
println("Grid size: $nlat × $nphi points")

# Transform: spectral → spatial
spatial = synthesize(cfg, sh)
println("Spatial field size: ", size(spatial))

free_config(cfg)
```

### Grid Types

SHTnsKit supports different grid layouts:

```julia
# Gauss-Legendre grid (optimal for spectral accuracy)
cfg_gauss = create_gauss_config(32, 32)
println("Gauss grid: $(get_nlat(cfg_gauss)) × $(get_nphi(cfg_gauss))")

# Regular equiangular grid  
cfg_regular = create_regular_config(32, 32)
println("Regular grid: $(get_nlat(cfg_regular)) × $(get_nphi(cfg_regular))")

free_config(cfg_gauss)
free_config(cfg_regular)
```

## Working with Real Data

### Creating Test Fields

```julia
cfg = create_gauss_config(24, 24)

# Get grid coordinates
θ, φ = get_coordinates(cfg)

# Create a test field: Y_2^1 (spherical harmonic)
test_field = real.(sqrt(15/(4π)) * sin(θ) .* cos(θ) .* exp.(1im * φ))

# Analyze to get spectral coefficients
sh_result = analyze(cfg, test_field)

# The coefficient for Y_2^1 should be ≈ 1.0
println("Y_2^1 coefficient: ", sh_result[get_index(cfg, 2, 1)])

free_config(cfg)
```

### Physical Fields

```julia
cfg = create_gauss_config(32, 32)
θ, φ = get_coordinates(cfg)

# Temperature-like field with equatorial maximum
temperature = 300 .+ 50 * cos.(2 * θ) .* cos.(φ)

# Transform to spectral domain
temp_sh = analyze(cfg, temperature)

# Reconstruct and compare
temp_reconstructed = synthesize(cfg, temp_sh)
reconstruction_error = norm(temperature - temp_reconstructed)
println("Temperature reconstruction error: $reconstruction_error")

free_config(cfg)
```

## Vector Fields

Vector fields on the sphere are decomposed into spheroidal and toroidal components:

```julia
cfg = create_gauss_config(20, 20)

# Create random spheroidal and toroidal coefficients
S_lm = rand(get_nlm(cfg))  # Spheroidal coefficients
T_lm = rand(get_nlm(cfg))  # Toroidal coefficients

# Synthesize vector field components
V_theta, V_phi = synthesize_vector(cfg, S_lm, T_lm)

println("Vector field size: ", size(V_theta), " and ", size(V_phi))

# Analyze back to get coefficients
S_recovered, T_recovered = analyze_vector(cfg, V_theta, V_phi)

# Check accuracy
S_error = norm(S_lm - S_recovered)
T_error = norm(T_lm - T_recovered)
println("Spheroidal error: $S_error, Toroidal error: $T_error")

free_config(cfg)
```

### Gradient and Curl

```julia
cfg = create_gauss_config(20, 20)

# Create a scalar potential
scalar_coeffs = rand(get_nlm(cfg))

# Compute gradient (gives a spheroidal vector field)
grad_theta, grad_phi = compute_gradient(cfg, scalar_coeffs)

# For a toroidal field, compute curl
toroidal_coeffs = rand(get_nlm(cfg))
curl_theta, curl_phi = compute_curl(cfg, toroidal_coeffs)

println("Gradient field size: ", size(grad_theta))
println("Curl field size: ", size(curl_theta))

free_config(cfg)
```

## Complex Fields

For complex-valued fields (e.g., wave functions):

```julia
cfg = create_gauss_config(16, 16)

# Create complex spectral coefficients
sh_complex = rand(ComplexF64, get_nlm(cfg))

# Complex field synthesis
spatial_complex = synthesize_complex(cfg, sh_complex)

# Complex field analysis
recovered_complex = analyze_complex(cfg, spatial_complex)

# Check accuracy
complex_error = norm(sh_complex - recovered_complex)
println("Complex field error: $complex_error")

free_config(cfg)
```

## Performance and Threading

### OpenMP Threading

```julia
# Check current thread settings
println("Current OpenMP threads: ", get_num_threads())

# Set thread count
set_num_threads(4)
println("Set to 4 threads: ", get_num_threads())

# Use optimal thread count for your system
set_optimal_threads()
println("Optimal threads: ", get_num_threads())
```

### Benchmarking

```julia
cfg = create_gauss_config(64, 64)
sh = rand(get_nlm(cfg))

# Time forward transform
@time spatial = synthesize(cfg, sh)

# Time backward transform  
@time recovered = analyze(cfg, spatial)

# Multiple runs for better statistics
println("Forward transform timing:")
@time for i in 1:10
    synthesize(cfg, sh)
end

free_config(cfg)
```

## GPU Acceleration

If you have CUDA available:

```julia
using CUDA

# Check if CUDA is functional
if CUDA.functional()
    # Create GPU-enabled configuration
    cfg_gpu = create_gpu_config(32, 32)
    
    # GPU transforms
    sh = rand(get_nlm(cfg_gpu))
    sh_gpu = CuArray(sh)
    
    spatial_gpu = synthesize_gpu(cfg_gpu, sh_gpu)
    println("GPU spatial field size: ", size(spatial_gpu))
    
    free_config(cfg_gpu)
else
    println("CUDA not available")
end
```

## Common Patterns

### In-Place Operations

For memory efficiency:

```julia
cfg = create_gauss_config(24, 24)

# Pre-allocate arrays
sh = allocate_spectral(cfg)
spatial = allocate_spatial(cfg)

# In-place operations (no additional allocation)
rand!(sh)
synthesize!(cfg, sh, spatial)  # spatial = synthesize(cfg, sh)
analyze!(cfg, spatial, sh)     # sh = analyze(cfg, spatial)

free_config(cfg)
```

### Batch Processing

```julia
cfg = create_gauss_config(20, 20)

# Process multiple fields
n_fields = 100
results = []

for i in 1:n_fields
    # Generate field
    sh = rand(get_nlm(cfg))
    
    # Process
    spatial = synthesize(cfg, sh)
    
    # Store result (example: compute mean)
    push!(results, mean(spatial))
    
    # Progress indicator
    i % 20 == 0 && println("Processed $i/$n_fields fields")
end

println("Mean of field means: ", mean(results))
free_config(cfg)
```

## Error Handling

```julia
cfg = create_gauss_config(16, 16)

try
    # Wrong array size
    wrong_sh = rand(10)  # Should be get_nlm(cfg)
    spatial = synthesize(cfg, wrong_sh)
catch e
    println("Caught expected error: ", e)
end

# Proper size check
sh = rand(get_nlm(cfg))
@assert length(sh) == get_nlm(cfg) "Wrong spectral array size"

spatial = synthesize(cfg, sh)
println("Successful transform with proper size")

free_config(cfg)
```

## Next Steps

Now that you've mastered the basics:

1. **Read the [API Reference](api/index.md)** for complete function documentation
2. **Explore [Examples](examples/index.md)** for real-world applications  
3. **Check [Performance Guide](performance.md)** for optimization tips
4. **See [Advanced Usage](advanced.md)** for complex workflows

## Quick Reference

```julia
# Configuration
cfg = create_gauss_config(lmax, mmax)
cfg = create_regular_config(lmax, mmax)

# Basic transforms
spatial = synthesize(cfg, spectral)
spectral = analyze(cfg, spatial)

# Vector transforms  
Vθ, Vφ = synthesize_vector(cfg, S_lm, T_lm)
S_lm, T_lm = analyze_vector(cfg, Vθ, Vφ)

# Complex fields
spatial_c = synthesize_complex(cfg, spectral_c)
spectral_c = analyze_complex(cfg, spatial_c)

# Threading
set_num_threads(n)
set_optimal_threads()

# Cleanup
free_config(cfg)
```