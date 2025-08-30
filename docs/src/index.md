# SHTnsKit.jl

High-performance spherical harmonic transforms for scientific computing

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/SHTnsKit.jl/stable)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

SHTnsKit.jl is a high-performance native Julia implementation of spherical harmonic transforms (SHT). It provides fast and memory‑efficient scalar, vector, and complex transforms with comprehensive parallel computing support, suitable for fluid dynamics, geophysics, astrophysics, and climate science applications.

## Features

### Core Transforms
- **Scalar Transforms**: Forward and backward spherical harmonic transforms
- **Complex Field Transforms**: Support for complex-valued fields on the sphere  
- **Vector Transforms**: Spheroidal-toroidal decomposition of vector fields
- **In-place Operations**: Memory-efficient transform operations

### Advanced Capabilities
- **Multiple Grid Types**: Gauss-Legendre and regular (equiangular) grids
- **Field Rotations**: Wigner D-matrix rotations in spectral and spatial domains
- **Power Spectrum Analysis**: Energy distribution across spherical harmonic modes
- **Multipole Analysis**: Expansion coefficients for gravitational/magnetic fields

### High-Performance Computing
- **MPI Parallelization**: Distributed spherical harmonic transforms with domain decomposition
- **SIMD Optimization**: Advanced vectorization with LoopVectorization.jl
- **Threading Controls**: Julia `Threads.@threads` loops and FFTW thread tuning
- **Memory Management**: Efficient allocation and thread‑safe operations
- **Automatic Differentiation**: Full support for ForwardDiff.jl and ChainRulesCore.jl

## Quick Start

### Serial Usage
```julia
using SHTnsKit

# Create spherical harmonic configuration
lmax = 32
cfg = create_gauss_config(lmax, lmax)

# Generate bandlimited test data (avoids high-frequency errors)
sh_coeffs = zeros(cfg.nlm)
sh_coeffs[1] = 1.0  # Y_0^0 constant term
if cfg.nlm > 3
    sh_coeffs[3] = 0.5  # Y_1^0 term
end

# Forward transform: spectral → spatial
spatial_field = synthesis(cfg, sh_coeffs)

# Backward transform: spatial → spectral
recovered_coeffs = analysis(cfg, spatial_field)

# Clean up
destroy_config(cfg)
```

### Parallel Usage (MPI)
```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()
cfg = create_gauss_config(20, 16; mres=48, nlon=64)
pcfg = create_parallel_config(cfg, COMM_WORLD)

# Parallel operations
sh_coeffs = randn(Complex{Float64}, cfg.nlm)
result = similar(sh_coeffs)
parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)

MPI.Finalize()
```

## Installation

### Basic Installation
```julia
using Pkg
Pkg.add("SHTnsKit")
```

### With Parallel Computing Support
```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

See the [Installation Guide](installation.md) for detailed setup instructions and MPI configuration.

## Documentation Overview

```@contents
Pages = [
    "installation.md",
    "parallel_installation.md",
    "quickstart.md", 
    "api/index.md",
    "examples/index.md",
    "performance.md",
    "advanced.md"
]
Depth = 2
```

## Scientific Applications

- **Fluid Dynamics**: Vorticity-divergence decomposition, stream function computation
- **Geophysics**: Gravitational and magnetic field analysis, Earth's surface modeling
- **Astrophysics**: Cosmic microwave background analysis, stellar surface dynamics
- **Climate Science**: Atmospheric and oceanic flow patterns, weather prediction
- **Plasma Physics**: Magnetohydrodynamics simulations, fusion plasma modeling

## Performance

SHTnsKit.jl achieves exceptional performance through:
- **MPI Parallelization**: Distributed computing with 2D domain decomposition
- **SIMD Vectorization**: Advanced optimizations with LoopVectorization.jl
- **Pure Julia kernels**: SIMD‑friendly loops with automatic optimization
- **FFTW integration**: Parallel FFTs with configurable threading
- **Memory efficiency**: Minimal allocations and optimized data layouts

| Problem Size (nlm) | Serial | 4 Processes | 16 Processes | Speedup |
|--------------------|--------|-------------|--------------|----------|
| 1,000             | 5ms    | 4ms         | 5ms          | 1.3x     |
| 10,000            | 50ms   | 18ms        | 12ms         | 4.2x     |
| 100,000           | 500ms  | 140ms       | 65ms         | 7.7x     |

## Citation

If you use SHTnsKit.jl in your research, please cite the package (citation info forthcoming).

## License

SHTnsKit.jl is released under the GNU General Public License v3.0, ensuring compatibility with the underlying SHTns library's CeCILL license.

## Support

- **Documentation**: Complete API reference and examples
- **Examples**: Comprehensive example gallery covering all use cases
- **Issues**: Report bugs and feature requests on [GitHub](https://github.com/subhk/SHTnsKit.jl/issues)
- **Discussions**: Community support and questions

## Related Packages

- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) - Fast Fourier transforms
- [SphericalHarmonics.jl](https://github.com/JuliaApproximation/SphericalHarmonics.jl) - Alternative pure Julia implementation
- [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl) - Various fast transforms
