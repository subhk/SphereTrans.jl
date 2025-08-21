# SHTnsKit.jl

Pure Julia spherical harmonic transforms for scientific computing

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/SHTnsKit.jl/stable)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

SHTnsKit.jl is a native Julia implementation of spherical harmonic transforms (SHT). It provides fast and memory‑efficient scalar, vector, and complex transforms without external C dependencies, suitable for fluid dynamics, geophysics, astrophysics, and climate science.

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
- **Threading Controls**: Julia `Threads.@threads` loops and FFTW thread tuning
- **Vectorization**: Leverages Julia/LLVM auto‑vectorization and FFTW
- **Memory Management**: Efficient allocation and thread‑safe operations

### Distributed Computing
- Not required. Focused on single‑process performance with Julia threads

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
destroy_config(cfg)
```

## Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

See the [Installation Guide](installation.md) for detailed setup instructions.

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

SHTnsKit.jl achieves strong performance through:
- Pure Julia kernels with SIMD‑friendly loops
- FFTW‑backed azimuthal transforms with configurable threading
- Memory‑efficient algorithms with minimal allocations

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
