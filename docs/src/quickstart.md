# Quick Start Guide

This tutorial will get you up and running with SHTnsKit.jl for spherical harmonic transforms in just a few minutes.

**What you'll learn:**
- Basic concepts: spectral vs spatial domains
- How to perform your first transform
- Working with real geophysical data
- Vector field analysis
- Performance optimization

**Prerequisites:** Basic Julia knowledge and familiarity with arrays and functions.

## Your First Transform

Let's start with a simple example to understand the basic workflow:

```julia
using SHTnsKit

# Step 1: Create a spherical harmonic configuration
lmax = 32        # Maximum degree (controls resolution)
mmax = 32        # Maximum order (typically same as lmax)
cfg = create_gauss_config(lmax, mmax)

# Step 2: Generate some random spectral coefficients
# These represent the "recipe" for building a function on the sphere
sh_coeffs = rand(get_nlm(cfg))
println("Number of coefficients: ", length(sh_coeffs))

# Step 3: Transform from spectral to spatial domain (synthesis)
# This builds the actual function values on a grid
spatial_field = synthesize(cfg, sh_coeffs)
println("Spatial field size: ", size(spatial_field))

# Step 4: Transform back to spectral domain (analysis)
# This recovers the coefficients from the spatial data
recovered_coeffs = analyze(cfg, spatial_field)

# Step 5: Check accuracy (should be very small)
error = norm(sh_coeffs - recovered_coeffs)
println("Round-trip error: $error")

# Step 6: Always clean up
destroy_config(cfg)
```

**What just happened?**
1. **Configuration**: We set up the transform parameters (resolution and grid type)
2. **Coefficients**: Created random spherical harmonic coefficients 
3. **Synthesis**: Converted coefficients → spatial values (spectral to physical)
4. **Analysis**: Converted spatial values → coefficients (physical to spectral)
5. **Verification**: The tiny error confirms the transforms are working correctly

## Understanding the Basics

### Spectral vs Spatial Domains

Understanding the two ways to represent data is key to using spherical harmonics effectively:

- **Spatial Domain**: Values at specific points on the sphere
  - Like having temperature measurements at weather stations
  - 2D array: `field[latitude, longitude]`
  - Easy to visualize and interpret physically

- **Spectral Domain**: Coefficients of mathematical basis functions (spherical harmonics)
  - Like having the "recipe" ingredients for recreating the field
  - 1D array: `coeffs[mode_index]`
  - Compact representation, efficient for analysis

**Analogy**: Think of a recipe vs a finished dish
- **Spatial** = the finished dish (what you see/taste)
- **Spectral** = the recipe (ingredients that make the dish)

```julia
cfg = create_gauss_config(16, 16)

# Spectral domain: 1D array of coefficients
nlm = get_nlm(cfg)        # Number of (l,m) coefficients  
sh = zeros(nlm)           # Initialize spectral coefficients
sh[1] = 1.0               # Set Y_0^0 = constant field (global average)
println("Spectral domain: ", length(sh), " coefficients")

# Spatial domain: 2D array of values on sphere
nlat, nphi = get_nlat(cfg), get_nphi(cfg)
println("Spatial domain: $nlat × $nphi = $(nlat*nphi) grid points")

# Transform: spectral → spatial (synthesis)
spatial = synthesize(cfg, sh)
println("Result: all values should be the same (constant field)")
println("Min/max values: ", extrema(spatial))

destroy_config(cfg)
```

**Key insight**: Setting only the first coefficient (`sh[1]`) creates a perfectly constant field over the entire sphere, demonstrating how spherical harmonics work as building blocks.

### Grid Types

SHTnsKit supports different ways to arrange points on the sphere. Think of it like choosing between different types of graph paper:

```julia
# Gauss-Legendre grid (optimal for spectral accuracy)
cfg_gauss = create_gauss_config(32, 32)
println("Gauss grid: $(get_nlat(cfg_gauss)) × $(get_nphi(cfg_gauss))")

# Regular equiangular grid  
cfg_regular = create_regular_config(32, 32)
println("Regular grid: $(get_nlat(cfg_regular)) × $(get_nphi(cfg_regular))")

destroy_config(cfg_gauss)
destroy_config(cfg_regular)
```

**Which grid should you use?**

- **Gauss-Legendre grid** (`create_gauss_config`):
  - **Best for**: Most scientific applications
  - **Pros**: Optimal mathematical properties, highest accuracy
  - **Cons**: Uneven spacing (denser near poles)
  - **Use when**: You want the best accuracy and don't need uniform spacing

- **Regular grid** (`create_regular_config`):
  - **Best for**: Visualization, interfacing with other software  
  - **Pros**: Uniform spacing, easier to understand
  - **Cons**: Slightly less accurate
  - **Use when**: You need uniform spacing or are working with external data

## Working with Real Data

### Creating Test Fields

```julia
cfg = create_gauss_config(24, 24)

# Get grid coordinate matrices
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a test field: Y_2^1 (illustrative)
test_field = real.(sqrt(15/(4π)) * sin.(θ) .* cos.(θ) .* exp.(1im .* φ))

# Analyze to get spectral coefficients
sh_result = analyze(cfg, test_field)

# The coefficient index for (l=2, m=1)
println("Y_2^1 coefficient: ", sh_result[lmidx(cfg, 2, 1)])

destroy_config(cfg)
```

### Physical Fields

```julia
cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Temperature-like field with equatorial maximum
temperature = 300 .+ 50 * cos.(2 * θ) .* cos.(φ)

# Transform to spectral domain
temp_sh = analyze(cfg, temperature)

# Reconstruct and compare
temp_reconstructed = synthesize(cfg, temp_sh)
reconstruction_error = norm(temperature - temp_reconstructed)
println("Temperature reconstruction error: $reconstruction_error")

destroy_config(cfg)
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

destroy_config(cfg)
```

### Gradient and Curl

```julia
cfg = create_gauss_config(20, 20)

# Example: compute spatial derivatives via FFT in φ
spatial = rand(get_nlat(cfg), get_nphi(cfg))
dφ = SHTnsKit.spatial_derivative_phi(cfg, spatial)

println("Spatial derivative field size: ", size(dφ))

destroy_config(cfg)
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

destroy_config(cfg)
```

## Performance and Threading

### Threading and FFTW threads

```julia
# Enable parallel loops and set FFTW threads sensibly
summary = set_optimal_threads!()
println(summary)  # (threads=..., fft_threads=...)

# Fine-tune
set_threading!(true)           # enable/disable parallel loops
set_fft_threads(4); get_fft_threads()
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

destroy_config(cfg)
```

## GPU Acceleration

This package is CPU‑focused and does not include GPU support.

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

destroy_config(cfg)
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
destroy_config(cfg)
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

destroy_config(cfg)
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
set_threading!(true)
set_optimal_threads!()

# Cleanup
destroy_config(cfg)
```
