# SHTnsKit.jl

*High-performance Julia wrapper for the SHTns C library*

[![Build Status](https://github.com/username/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/username/SHTnsKit.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://username.github.io/SHTnsKit.jl/stable)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

SHTnsKit.jl provides a comprehensive Julia interface to the high-performance [SHTns](https://nschaeff.bitbucket.io/shtns/) (Spherical Harmonic Transform) library. It enables fast and efficient spherical harmonic transforms for scientific computing applications in fluid dynamics, geophysics, astrophysics, and climate science.

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

### Performance Optimizations
- **OpenMP Multi-threading**: Automatic detection and optimal thread configuration
- **GPU Acceleration**: CUDA support with host-device memory management
- **Vectorization**: Support for SSE, AVX, and other SIMD instruction sets
- **Memory Management**: Efficient allocation and thread-safe operations

### Distributed Computing
- **MPI Support**: Distributed transforms across multiple nodes
- **Hybrid Parallelism**: Combined MPI + OpenMP + GPU acceleration
- **Scalable**: Efficient scaling to large computational clusters

## Quick Start

```julia
using SHTnsKit

# Create spherical harmonic configuration
lmax = 32
cfg = create_gauss_config(lmax, lmax)

# Generate test data
sh_coeffs = rand(get_nlm(cfg))

# Forward transform: spectral → spatial
spatial_field = synthesize(cfg, sh_coeffs)

# Backward transform: spatial → spectral
recovered_coeffs = analyze(cfg, spatial_field)

# Clean up
free_config(cfg)
```

## Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

See the [Installation Guide](installation.md) for detailed setup instructions including SHTns C library installation.

## Documentation Overview

```@contents
Pages = [
    "installation.md",
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
- Direct interface to optimized SHTns C library
- Automatic SIMD vectorization (SSE, AVX, AVX-512)
- OpenMP parallelization with optimal thread management
- GPU acceleration for large-scale problems
- Memory-efficient operations with minimal allocations

## Citation

If you use SHTnsKit.jl in your research, please cite:

**SHTns library**: Schaeffer, N. (2013). Efficient Spherical Harmonic Transforms aimed at pseudospectral numerical simulations. *Geochemistry, Geophysics, Geosystems*, 14(3), 751-758.

**SHTnsKit.jl**: [Citation will be provided upon publication]

## License

SHTnsKit.jl is released under the GNU General Public License v3.0, ensuring compatibility with the underlying SHTns library's CeCILL license.

## Support

- **Documentation**: Complete API reference and examples
- **Examples**: Comprehensive example gallery covering all use cases
- **Issues**: Report bugs and feature requests on [GitHub](https://github.com/username/SHTnsKit.jl/issues)
- **Discussions**: Community support and questions

## Related Packages

- [FFTW.jl](https://github.com/JuliaMath/FFTW.jl) - Fast Fourier transforms
- [SphericalHarmonics.jl](https://github.com/JuliaApproximation/SphericalHarmonics.jl) - Pure Julia implementation
- [FastTransforms.jl](https://github.com/JuliaApproximation/FastTransforms.jl) - Various fast transforms