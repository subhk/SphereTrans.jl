# SHTnsKit.jl

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)

<!-- Badges -->
 <p align="left">
    <a href="https://subhk.github.io/SHTnsKit.jl">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20-blue">
    </a>
      <a href="https://subhk.github.io/SHTnsKit.jl">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-dev%20-orange">
    </a>
</p>

**Fast spherical harmonic transforms for Julia**

SHTnsKit.jl provides a comprehensive, pure-Julia interface for spherical harmonic transforms, enabling efficient analysis of functions on the sphere. Whether you're working with climate data, geophysical fields, or astrophysical simulations, this package offers the tools you need for spectral analysis and synthesis.

**What are spherical harmonics?** Think of them as "Fourier transforms for the sphere" - they decompose any function on a spherical surface into a series of mathematical basis functions, just like how Fourier transforms decompose signals into frequencies.

## Why SHTnsKit.jl?

- **Pure Julia**: No C dependencies, integrates seamlessly with the Julia ecosystem
- **Complete functionality**: Scalar, vector, and complex field transforms
- **Scientific accuracy**: Multiple grid types and normalization conventions
- **High performance**: Optimized with Julia threads and FFTW
- **Easy to use**: Clear API with comprehensive examples and documentation

## Key Features

### Transform Types
- **Scalar Fields**: Convert between spatial values and spherical harmonic coefficients
- **Vector Fields**: Decompose vector fields into divergent (spheroidal) and rotational (toroidal) components
- **Complex Fields**: Full support for complex-valued functions on the sphere

### Grid Support
- **Gauss-Legendre grids**: Optimal for spectral accuracy (recommended)
- **Regular grids**: Equiangular spacing for specific applications

### Advanced Capabilities
- **Power spectrum analysis**: Understand energy distribution across scales
- **Field rotations**: Rotate functions using Wigner D-matrices
- **Automatic differentiation**: Seamless integration with ForwardDiff.jl and Zygote.jl
- **High performance**: Multi-threading support with Julia threads and FFTW

### Scientific Applications
- **Climate science**: Analyze atmospheric and oceanic fields
- **Geophysics**: Model gravitational and magnetic fields
- **Astrophysics**: Study stellar surfaces and cosmic microwave background
- **Fluid dynamics**: Decompose velocity fields and compute vorticity



## Installation

SHTnsKit.jl is a pure Julia package with no external dependencies. Install it using the Julia package manager:

```julia
using Pkg
Pkg.add("SHTnsKit")
```

That's it! No additional system libraries or compilation required.

## Quick Start

Here's a minimal example to get you started:

```julia
using SHTnsKit

# Create a spherical harmonic configuration
lmax = 16              # Maximum spherical harmonic degree
cfg = create_gauss_config(lmax, lmax)

# Create some test data on the sphere
spatial_data = rand(get_nlat(cfg), get_nphi(cfg))

# Transform to spherical harmonic coefficients
coeffs = analyze_real(cfg, spatial_data)

# Transform back to spatial domain
reconstructed = synthesize_real(cfg, coeffs)

# Check accuracy
error = maximum(abs.(spatial_data - reconstructed))
println("Reconstruction error: $error")

# Clean up
destroy_config(cfg)
```

**What just happened?**
1. We created a configuration for spherical harmonics up to degree 16
2. We generated random data on a sphere (like temperature measurements)
3. We decomposed this data into spherical harmonic modes
4. We reconstructed the original data from these modes
5. The tiny error shows the transform is working correctly!

## Learning Path

**New to spherical harmonics?** Start with these examples in order:

### 1. Understanding the Basics

```julia
using SHTnsKit

# Create a configuration
cfg = create_gauss_config(16, 16)

# Get grid coordinates
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a simple pattern: hot equator, cold poles
temperature = @. 300 + 50 * cos(θ)  # 300K base + 50K variation

# Transform to spherical harmonics
coeffs = analyze_real(cfg, temperature)
println("Number of coefficients: ", length(coeffs))

# Find the dominant mode
max_idx = argmax(abs.(coeffs))
println("Strongest coefficient at index: $max_idx")

# Reconstruct and verify
reconstructed = synthesize_real(cfg, coeffs)
error = maximum(abs.(temperature - reconstructed))
println("Reconstruction error: $error")

destroy_config(cfg)
```

**Key concepts:**
- `θ` (theta): colatitude (0 at north pole, π at south pole)
- `φ` (phi): longitude (0 to 2π)
- Each coefficient represents how much of a specific spherical harmonic pattern is present
- The reconstruction should be nearly perfect (tiny numerical error)

### 2. Working with Real Data

```julia
using SHTnsKit

# Create configuration
cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a realistic temperature field with seasonal variation
# Simulate July temperatures (summer in Northern Hemisphere)
july_temps = @. 273.15 + 40 * cos(θ - 0.4)  # Shifted toward NH summer

# Transform to spectral domain
temp_coeffs = analyze_real(cfg, july_temps)

# Compute power spectrum to see which scales dominate
power = power_spectrum(cfg, temp_coeffs)

# Print the first few modes
for l in 0:5
    println("Degree l=$l power: ", power[l+1])
end

# The l=0 mode is the global mean
global_mean = temp_coeffs[1] / sqrt(4π)  # Normalized
println("Global mean temperature: $global_mean K")

destroy_config(cfg)
```

**What this shows:**
- How to create realistic geophysical data
- Power spectrum analysis reveals which spatial scales are important
- The l=0 mode gives you the global average
- Higher l modes represent finer spatial details

### 3. Vector Field Analysis

```julia
using SHTnsKit

# Create configuration
cfg = create_gauss_config(24, 24)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a realistic wind field: jet stream + tropical circulation
# Zonal (east-west) component
u = @. 20 * sin(2*θ) + 5 * cos(3*φ) * sin(θ)
# Meridional (north-south) component  
v = @. 10 * cos(θ) * sin(2*φ)

# Analyze vector field using real-basis (easier to understand)
S_coeffs, T_coeffs = analyze_vector_real(cfg, u, v)

# Compute energy in divergent vs rotational components
divergent_energy = sum(abs2, S_coeffs)
rotational_energy = sum(abs2, T_coeffs)
total_energy = divergent_energy + rotational_energy

println("Flow analysis:")
println("  Divergent (spheroidal): $(100*divergent_energy/total_energy)%")
println("  Rotational (toroidal): $(100*rotational_energy/total_energy)%")

# Reconstruct and verify
u_reconstructed, v_reconstructed = synthesize_vector_real(cfg, S_coeffs, T_coeffs)
u_error = maximum(abs.(u - u_reconstructed))
v_error = maximum(abs.(v - v_reconstructed))
println("  Reconstruction error: u=$u_error, v=$v_error")

destroy_config(cfg)
```

**Physical interpretation:**
- **Spheroidal (S) modes**: Represent divergent flow (expansion/compression)
- **Toroidal (T) modes**: Represent rotational flow (circulation, vortices)
- Real atmospheric flows are usually dominated by rotational motion
- This decomposition is fundamental in meteorology and oceanography

### 4. Advanced Analysis: Power Spectra

```julia
using SHTnsKit
using Plots

# Create configuration
cfg = create_gauss_config(64, 64)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a multi-scale field (like turbulent flow)
# Large-scale + medium-scale + small-scale components
field = @. (sin(2*θ) * cos(φ) +           # Large scale (l~2)
            0.3 * sin(6*θ) * cos(3*φ) +   # Medium scale (l~6) 
            0.1 * sin(15*θ) * cos(8*φ))   # Small scale (l~15)

# Transform to spectral domain
coeffs = analyze_real(cfg, field)

# Compute power spectrum
power = power_spectrum(cfg, coeffs)

# Plot to see the energy distribution
plot(0:get_lmax(cfg), power, 
     xlabel="Spherical Harmonic Degree l", 
     ylabel="Power", yscale=:log10,
     title="Energy Spectrum", linewidth=2)

# Find the peak energy scale
max_power_idx = argmax(power[2:end]) + 1  # Skip l=0
println("Peak energy at degree l = $(max_power_idx-1)")
println("This corresponds to ~$(360/(max_power_idx-1))° wavelength")

destroy_config(cfg)
```

**Understanding power spectra:**
- Each degree l corresponds to a spatial wavelength ~360°/l
- Power spectrum shows how energy is distributed across scales
- Peaks indicate dominant spatial scales in your data
- Useful for understanding the characteristic sizes of features

## Real-World Applications

### Climate Data Analysis

```julia
using SHTnsKit

# Process temperature anomalies
cfg = create_gauss_config(32, 32)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Simulate El Niño pattern (Pacific warming)
temp_anomaly = @. 2 * exp(-((φ-π)^2 + (θ-π/2)^2) / 0.3^2)

# Analyze the spatial pattern
coeffs = analyze_real(cfg, temp_anomaly)
power = power_spectrum(cfg, coeffs)

# Find characteristic scale
max_l = argmax(power[2:end])  # Skip l=0
characteristic_scale = 360 / max_l
println("El Niño characteristic scale: ~$characteristic_scale degrees")

destroy_config(cfg)
```

### Geophysical Modeling

```julia
# Gravitational field analysis
cfg = create_gauss_config(48, 48)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Earth's oblateness (J₂ term) + smaller harmonics
gravity_anomaly = @. -9.81 * 0.001 * (1.5*cos(θ)^2 - 0.5) + 
                     -9.81 * 0.0001 * sin(3*θ) * cos(2*φ)

coeffs = analyze_real(cfg, gravity_anomaly)

# The J₂ coefficient (degree 2, order 0)
J2_index = SHTnsKit.lmidx(cfg, 2, 0)
J2_value = coeffs[J2_index]
println("J₂ coefficient: $J2_value")

destroy_config(cfg)
```

## Advanced Features

### Rotations and Coordinate Transformations

```julia
using SHTnsKit

# Create a field with a specific pattern
cfg = create_gauss_config(16, 16)
θ, φ = SHTnsKit.create_coordinate_matrices(cfg)

# Create a field concentrated at north pole
original_field = @. exp(-5 * θ^2)  # Hot spot at north pole

# Transform to coefficients
coeffs = analyze_real(cfg, original_field)

# Rotate by 45° around z-axis (longitude shift)
rotate_real!(cfg, coeffs; alpha=π/4, beta=0.0, gamma=0.0)

# Transform back to see the rotated field
rotated_field = synthesize_real(cfg, coeffs)

# The hot spot should now be at longitude 45°
println("Original max at: ", argmax(original_field))
println("Rotated max at: ", argmax(rotated_field))

destroy_config(cfg)
```

**Rotation applications:**
- Change coordinate systems (e.g., from geographic to magnetic coordinates)
- Align data from different observation times
- Study rotational symmetries in your data

### Performance Optimization

```julia
using SHTnsKit

# Enable threading for better performance
set_optimal_threads!()  # Automatically configure threads

# For manual control:
# set_threading!(true)        # Enable Julia thread parallelization
# set_fft_threads(4)         # Set FFTW threads

# Create larger configuration for benchmarking
cfg = create_gauss_config(64, 64)
field = rand(get_nlat(cfg), get_nphi(cfg))

# Time the transforms
@time coeffs = analyze_real(cfg, field)      # Analysis
@time reconstructed = synthesize_real(cfg, coeffs)  # Synthesis

destroy_config(cfg)
```

## Automatic Differentiation

SHTnsKit.jl provides comprehensive automatic differentiation support through extensions for both **ForwardDiff.jl** (forward-mode) and **Zygote.jl** (reverse-mode), enabling gradient-based optimization and machine learning applications.

### Forward-mode AD (ForwardDiff.jl)

```julia
using SHTnsKit, ForwardDiff

cfg = create_gauss_config(8, 8)
sh_coeffs = rand(get_nlm(cfg))

# Define objective function
function total_power(sh)
    spatial = synthesize(cfg, sh)
    return sum(abs2, spatial)
end

# Compute gradient
gradient = ForwardDiff.gradient(total_power, sh_coeffs)
hessian = ForwardDiff.hessian(total_power, sh_coeffs)
```

### Reverse-mode AD (Zygote.jl)

```julia
using SHTnsKit, Zygote

# Same function as above
function loss_function(sh)
    spatial = synthesize(cfg, sh)
    return sum(abs2, spatial)
end

# Get both value and gradient
value, gradient = Zygote.withgradient(loss_function, sh_coeffs)
```

### Optimization Example

```julia
# Target fitting with gradient descent
target_field = create_test_field(cfg, 2, 1)  # Y_2^1 harmonic

function mse_loss(sh_coeffs)
    predicted = synthesize(cfg, sh_coeffs) 
    return sum((predicted - target_field).^2) / length(target_field)
end

# Initialize and optimize
sh_coeffs = 0.1 * randn(get_nlm(cfg))
learning_rate = 0.01

for i in 1:100
    loss_val, grad = Zygote.withgradient(mse_loss, sh_coeffs)
    sh_coeffs .-= learning_rate .* grad[1]  # Gradient descent step
end
```

### Supported Functions

All major SHTnsKit functions support automatic differentiation:
- `synthesize`, `analyze` - Core transforms
- `synthesize_vector`, `analyze_vector` - Vector field transforms  
- `evaluate_at_point` - Point evaluation
- `power_spectrum`, `total_power` - Spectral analysis
- `spatial_integral`, `spatial_mean` - Spatial operations

See `docs/automatic_differentiation.md` for comprehensive examples and `examples/differentiation_examples.jl` for runnable code.
Additional runnable examples are provided in `examples/ad_examples.jl` showing both Zygote and ForwardDiff on small problems.

### Vector optimization with Zygote (example)

```julia
using SHTnsKit, Zygote, Random

cfg = create_gauss_config(8, 8)
n = length(SHTnsKit._cplx_lm_indices(cfg))
rng = MersenneTwister(42)

# Build a target vector field
S_tar = [0.3randn(rng) + 0.3im*randn(rng) for _ in 1:n]
T_tar = [0.3randn(rng) + 0.3im*randn(rng) for _ in 1:n]
uθ_tar, uφ_tar = cplx_synthesize_vector(cfg, S_tar, T_tar)

# Loss
loss(S, T) = begin
    uθ, uφ = cplx_synthesize_vector(cfg, S, T)
    0.5 * (sum(abs2, uθ .- uθ_tar) + sum(abs2, uφ .- uφ_tar))
end

# Optimize S,T to match the target
S = zeros(ComplexF64, n); T = zeros(ComplexF64, n)
for it in 1:20
    L, back = Zygote.pullback(loss, S, T)
    gS, gT = back(1.0)
    S .-= 0.1 .* gS
    T .-= 0.1 .* gT
    @show it L
end
destroy_config(cfg)
```

See also `examples/ad_vector_zygote.jl` for a complete runnable script.

## Thread Safety

All SHTns operations are thread-safe when using different configurations. Operations on the same configuration are automatically serialized using per-config locks.

## Grid Types

SHTnsKit.jl supports multiple grid types:

```julia
# Gauss-Legendre grid (recommended for most applications)
cfg_gauss = create_gauss_config(32, 32)

# Regular (equiangular) grid
cfg_regular = create_regular_config(32, 32)

```

## Complex Fields

```julia
using SHTnsKit

cfg = create_gauss_config(16, 16)

# Create complex spectral coefficients
sh_complex = allocate_complex_spectral(cfg)
sh_complex[1] = 1.0 + 0.5im

# Transform to spatial domain
spatial_complex = synthesize_complex(cfg, sh_complex)

# Transform back
recovered_complex = analyze_complex(cfg, spatial_complex)

destroy_config(cfg)
```

## Error Handling

SHTnsKit.jl provides robust error handling:

```julia
using SHTnsKit

try
    cfg = create_gauss_config(64, 64)
    
    # Operations that might fail
    sh = rand(10)  # Wrong size
    spat = allocate_spatial(cfg)
    synthesize!(cfg, sh, spat)  # Will throw an error due to wrong size
    
catch e
    println("Error: $e")
finally
    if @isdefined(cfg)
        destroy_config(cfg)
    end
end
```

## Performance Tips

1. **Use Gauss grids** for most applications - they're more efficient
2. **Use in-place operations** (`synthesize!`, `analyze!`) when possible
3. **Batch operations** on the same configuration for better cache usage

## Benchmarking

```julia
using SHTnsKit
using BenchmarkTools

cfg = create_gauss_config(64, 64)
sh = rand(get_nlm(cfg))
spat = allocate_spatial(cfg)

@benchmark synthesize!($cfg, $sh, $spat)
```

### Real vs Complex Roundtrip (examples)

Run the example scripts to compare real-basis and complex roundtrip timings:

```bash
julia --project examples/compare_real_complex.jl
julia --project examples/compare_vector_real_complex.jl
julia --project examples/profile_complex.jl
```

These print average timings (after warmup) for a few (lmax, mmax) pairs.

### Threading Controls

SHTnsKit parallelizes selected loops with Julia threads and can thread FFTs via FFTW. Start Julia with threads and set FFT threads:

```julia
using SHTnsKit
using Base.Threads

println("Julia threads: ", nthreads())
set_threading!(true)              # enable parallel loops (default)
set_fft_threads(nthreads())       # use same thread count in FFTW
set_optimal_threads!()            # convenience helper
```

Note: OpenMP is not used directly; threading is pure Julia + FFTW. Avoid multiple concurrent transforms on the same `cfg` from different tasks.

Environment variables (optional):

- `SHTNSKIT_THREADS` = `1|true|yes|on` or `0|false|no|off` to enable/disable loop threading at startup.
- `SHTNSKIT_FFT_THREADS` = integer to set FFTW thread count at startup.
- `SHTNSKIT_AUTO_THREADS` = `1|true|yes|on` to call `set_optimal_threads!()` at startup.

Example (bash):

```bash
export JULIA_NUM_THREADS=$(nproc)
export SHTNSKIT_THREADS=on
export SHTNSKIT_FFT_THREADS=$(nproc)
export SHTNSKIT_AUTO_THREADS=on
julia --project -e 'using SHTnsKit; @show SHTnsKit.get_threading(); @show SHTnsKit.get_fft_threads()'
```

### Quick threading benchmark

```julia
using SHTnsKit
using Base.Threads

cfg = create_gauss_config(32, 32)

# single-thread style
set_threading!(false)
set_fft_threads(1)
t1 = @elapsed cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, allocate_complex_spectral(cfg)))

# multi-thread style
set_threading!(true)
set_fft_threads(nthreads())
t2 = @elapsed cplx_spat_to_sh(cfg, cplx_sh_to_spat(cfg, allocate_complex_spectral(cfg)))

println("speedup ≈ ", round(t1/max(t2,1e-12), digits=2), "x (Nthreads=", nthreads(), ")")
destroy_config(cfg)
```

For a more thorough comparison, run:

```bash
julia --project examples/benchmark_threading.jl
```

### Benchmark Results (fill with your numbers)

You can generate a Markdown table of timings with:

```bash
julia --project examples/record_benchmarks.jl
```

Then copy the output here for reference. A table template:

| lmax=mmax | Nthreads | synth ST (s) | synth MT (s) | speedup | analyze(syn) ST (s) | analyze(syn) MT (s) | speedup |
|-----------|----------|--------------|--------------|---------|---------------------|---------------------|---------|
| 24        |          |              |              |         |                     |                     |         |
| 32        |          |              |              |         |                     |                     |         |
| 40        |          |              |              |         |                     |                     |         |

## Next Steps

Now that you've seen the basics, here's how to dive deeper:

### **Documentation**
- **[Complete API Documentation](https://subhk.github.io/SHTnsKit.jl)**: Detailed function reference
- **[Advanced Examples](docs/src/examples/index.md)**: Real-world scientific applications
- **[Performance Guide](docs/src/performance.md)**: Optimization tips and benchmarking

### **Scientific Applications**
- **Climate Science**: Temperature anomaly analysis, seasonal cycles
- **Geophysics**: Gravitational fields, magnetic field modeling  
- **Astrophysics**: Cosmic microwave background, stellar surface analysis
- **Fluid Dynamics**: Vorticity-divergence decomposition, turbulence analysis

### **Advanced Features**
- **Automatic Differentiation**: Gradient-based optimization with ForwardDiff.jl/Zygote.jl
- **Vector Fields**: Spheroidal-toroidal decomposition for wind/flow analysis
- **Field Rotations**: Coordinate system transformations
- **Complex Fields**: Support for wave functions and complex-valued data

### **Getting Help**
- **Examples**: Run `julia examples/` scripts for hands-on learning
- **Issues**: Report bugs or ask questions on [GitHub Issues](https://github.com/subhk/SHTnsKit.jl/issues)
- **Discussions**: Community support for usage questions

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests on [GitHub](https://github.com/subhk/SHTnsKit.jl/issues).

## Citation

If you use SHTnsKit.jl in your research, please cite:

1. **SHTns library**: Schaeffer, N. (2013). Efficient Spherical Harmonic Transforms aimed at pseudospectral numerical simulations. *Geochemistry, Geophysics, Geosystems*, 14(3), 751-758.

2. 

## License

SHTnsKit.jl is released under the GNU General Public License v3.0 (GPL-3.0), the same license family as the underlying SHTns library which uses the CeCILL License (GPL-compatible). This ensures full compatibility and aligns with the open-source philosophy of the SHTns project.

## References

- [SHTns Official Documentation](https://nschaeff.bitbucket.io/shtns/)
- [SHTns Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/ggge.20071)
- [Spherical Harmonics on Wikipedia](https://en.wikipedia.org/wiki/Spherical_harmonics)
