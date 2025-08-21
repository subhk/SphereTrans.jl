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

A comprehensive Julia interface for spherical harmonic transforms inspired by [SHTns](https://nschaeff.bitbucket.io/shtns/), providing fast and efficient spherical harmonic transforms for scientific computing applications. This package is implemented purely in Julia (no C FFI required). Complex coefficients are the canonical representation, with a parity-accurate real-basis API built on top.

## Features

SHTnsKit.jl provides a complete interface to all SHTns features:

### Core Transforms
- **Complex Transforms (canonical)**: Analysis/synthesis for complex fields with full ±m modes
- **Real-Basis Parity API**: `analyze_real` and `synthesize_real` with correct cos/sin storage for m>0
- **Vector Transforms**: Complex spheroidal/toroidal analysis/synthesis with precise derivatives
- **In-place Operations**: Memory-efficient transform operations

### Analysis and Utilities
- **Multiple Grid Types**: Gauss-Legendre and regular (equiangular) grids
- **Power Spectrum Analysis**: Energy distribution across spherical harmonic modes
- **Spatial Operations**: Area-weighted integrals, means, variance, regridding
- **Rotation Support**: Rotate coefficients via Wigner-D (ZYZ Euler angles)
- **Automatic Differentiation**: ForwardDiff.jl and Zygote.jl support (via extensions)

### Notes
- GPU acceleration and explicit OpenMP threading controls are not enabled at this time.


## Installation

SHTnsKit.jl can be installed using the Julia package manager:

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Prerequisites

SHTnsKit.jl requires the SHTns C library to be installed on your system. You can install it using:

<!-- **Ubuntu/Debian:**
```bash
sudo apt-get install libshtns-dev
``` -->

<!-- **macOS (Homebrew):**
```bash
brew install shtns
``` -->

**From source:**
```bash
wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-3.x.x.tar.gz
tar -xzf shtns-3.x.x.tar.gz
cd shtns-3.x.x
./configure --enable-openmp --enable-ishioka --enable-magic-layout
make && sudo make install
```

### Custom Library Path

This package does not depend on a system `libshtns` and does not call into C.

## Quick Start

```julia
using SHTnsKit

# Create a spherical harmonic configuration
lmax = 16              # Maximum spherical harmonic degree
cfg = create_gauss_config(lmax, lmax)

# Create test data
nlm = get_nlm(cfg)     # Number of spectral coefficients
sh_coeffs = rand(nlm)  # Random spectral coefficients

# Forward transform: spectral → spatial (real-basis API)
rcoeffs = analyze_real(cfg, rand(get_nlat(cfg), get_nphi(cfg)))
spatial_field = synthesize_real(cfg, rcoeffs)

# Complex API (canonical)
sh_cplx = allocate_complex_spectral(cfg)
spatial_cplx = synthesize_complex(cfg, sh_cplx)
recovered_cplx = analyze_complex(cfg, spatial_cplx)

# Clean up
destroy_config(cfg)
```

## Examples

### Basic Scalar Transform

```julia
using SHTnsKit

# Set up configuration with Gauss-Legendre grid
cfg = create_gauss_config(32, 32)  # lmax=32, mmax=32
nlat, nphi = get_nlat(cfg), get_nphi(cfg)

# Create a test function: Y_2^1 spherical harmonic
function create_Y21(cfg)
    spat = zeros(get_nlat(cfg), get_nphi(cfg))
    for i in 1:get_nlat(cfg)
        theta = get_theta(cfg, i-1)
        for j in 1:get_nphi(cfg)
            phi = get_phi(cfg, j)
            spat[i, j] = sqrt(15/(8π)) * sin(theta) * cos(theta) * cos(phi)
        end
    end
    return spat
end

# Transform and analyze
spatial = create_Y21(cfg)
spectral = analyze(cfg, spatial)
reconstructed = synthesize(cfg, spectral)

println("Transform error: ", maximum(abs.(spatial - reconstructed)))
destroy_config(cfg)
```

### Vector Field Analysis (complex)

```julia
using SHTnsKit
using LinearAlgebra

# Create configuration
cfg = create_gauss_config(16, 16)

# Create a vector field (e.g., surface winds)
using Complexes: ComplexF64
u = rand(get_nlat(cfg), get_nphi(cfg))
v = rand(get_nlat(cfg), get_nphi(cfg))
Slm, Tlm = cplx_analyze_vector(cfg, ComplexF64.(u), ComplexF64.(v))

# Compute energy in each component
total_energy = sum(Slm.^2 + Tlm.^2)
spheroidal_fraction = sum(Slm.^2) / total_energy
toroidal_fraction = sum(Tlm.^2) / total_energy

println("Spheroidal (divergent) energy: $(spheroidal_fraction*100)%")
println("Toroidal (rotational) energy: $(toroidal_fraction*100)%")

# Reconstruct vector field
u_reconstructed, v_reconstructed = cplx_synthesize_vector(cfg, Slm, Tlm)

free_config(cfg)
```

### Power Spectrum Analysis

```julia
using SHTnsKit

cfg = create_gauss_config(64, 64)

# Create some test data
sh = rand(get_nlm(cfg))

# Compute power spectrum
power = power_spectrum(cfg, sh)

# Analyze spectral slope
using Plots
plot(0:get_lmax(cfg), power, 
     xlabel="Spherical Harmonic Degree l", 
     ylabel="Power",
     yscale=:log10, title="Energy Spectrum")

destroy_config(cfg)
```

### Rotations (ZYZ Euler angles)

```julia
cfg = create_gauss_config(12, 12)
coeffs = allocate_complex_spectral(cfg)
# rotate in-place by (alpha, beta, gamma)
rotate_complex!(cfg, coeffs; alpha=0.2, beta=0.3, gamma=0.1)

# Real-basis rotation
r = analyze_real(cfg, rand(get_nlat(cfg), get_nphi(cfg)))
rotate_real!(cfg, r; alpha=0.2, beta=0.3, gamma=0.1)
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
