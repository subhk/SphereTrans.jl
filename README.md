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

**High-Performance Spherical Harmonic Transforms for Julia**

SHTnsKit.jl provides a comprehensive, pure-Julia implementation of spherical harmonic transforms with **parallel computing support** for scalable scientific computing. From single-core laptops to large HPC clusters, this package delivers the performance you need for spectral analysis on the sphere.

## Key Features

### **High-Performance Computing**
- **Pure Julia**: No C dependencies, seamless Julia ecosystem integration
- **Multi-threading**: Optimized with Julia threads and FFTW parallelization
- **MPI Parallel**: Distributed computing with MPI + PencilArrays + PencilFFTs
- **SIMD Optimized**: Vectorization with LoopVectorization.jl support
- **Extensible**: Modular architecture for CPU/GPU/distributed computing

### **Complete Scientific Functionality**  
- **Transform Types**: Scalar, vector, and complex field transforms
- **Grid Support**: Gauss-Legendre and regular (equiangular) grids
- **Vector Analysis**: Spheroidal-toroidal decomposition for flow fields
- **Differential Operators**: Laplacian, gradient, divergence, vorticity
- **Spectral Analysis**: Power spectra, correlation functions, filtering

### **Advanced Capabilities**
- **Automatic Differentiation**: Native ForwardDiff.jl and Zygote.jl support  
- **Field Rotations**: Wigner D-matrix rotations and coordinate transforms
- **Matrix Operators**: Efficient spectral differential operators
- **Performance Tuning**: Comprehensive benchmarking and optimization tools


## Installation

### Basic Installation (Serial Computing)

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Full Installation (Parallel Computing)

For high-performance parallel computing on clusters:

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

### System Requirements

**MPI Setup** (for parallel computing):
```bash
# Ubuntu/Debian
sudo apt-get install libopenmpi-dev

# macOS  
brew install open-mpi

# Configure Julia MPI
julia -e 'using Pkg; Pkg.build("MPI")'
```

## Quick Start

### Basic Usage (Serial)

```julia
using SHTnsKit

# Create spherical harmonic configuration
lmax = 32              # Maximum spherical harmonic degree  
cfg = create_gauss_config(Float64, lmax, lmax, 2*lmax+2, 4*lmax+1)

# Create test data on the sphere
spatial_data = rand(get_nlat(cfg), get_nphi(cfg))

# Transform to spherical harmonic coefficients
coeffs = allocate_spectral(cfg)
spat_to_sh!(cfg, spatial_data, coeffs)

# Transform back to spatial domain  
reconstructed = allocate_spatial(cfg)
sh_to_spat!(cfg, coeffs, reconstructed)

# Check accuracy
error = maximum(abs.(spatial_data - reconstructed))
println("Roundtrip error: $error")  # Should be ~1e-14

destroy_config(cfg)
```

### Parallel Computing (MPI)

```julia
using SHTnsKit
using MPI, PencilArrays, PencilFFTs

MPI.Init()

# Create configuration
cfg = create_gauss_config(Float64, 64, 64, 130, 256)

# Create parallel configuration for MPI
pcfg = create_parallel_config(cfg, MPI.COMM_WORLD)

# Parallel spherical harmonic operations
sh_coeffs = randn(ComplexF64, cfg.nlm)
result = similar(sh_coeffs)

# Parallel operators
parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)  # No communication
parallel_apply_operator(pcfg, :costheta, sh_coeffs, result)   # Requires communication

# Parallel transforms
spatial_data = allocate_spatial(cfg)
memory_efficient_parallel_transform!(pcfg, :synthesis, sh_coeffs, spatial_data)

MPI.Finalize()
```

### High-Performance SIMD

```julia
using SHTnsKit
using LoopVectorization  # Enables turbo optimizations

cfg = create_gauss_config(Float64, 48, 48, 98, 192)
sh_coeffs = randn(ComplexF64, cfg.nlm)

# Turbo-optimized operations (when LoopVectorization is available)
turbo_apply_laplacian!(cfg, sh_coeffs)

# Benchmark SIMD vs regular implementations
results = benchmark_turbo_vs_simd(cfg)
println("SIMD speedup: $(results.speedup)x")

destroy_config(cfg)
```

## Comprehensive Examples

### 1. Climate Data Analysis

```julia
using SHTnsKit

# Setup for climate-scale problem
cfg = create_gauss_config(Float64, 42, 42, 86, 128)  # ~2.8° resolution
θ, φ = get_theta(cfg), get_phi(cfg)

# Create realistic temperature field with seasonal variation
summer_pattern = @. 273.15 + 40 * cos(θ - 0.4) + 10 * sin(2*φ) * sin(θ)
coeffs = allocate_spectral(cfg)
spat_to_sh!(cfg, summer_pattern, coeffs)

# Analyze dominant spatial scales
power = power_spectrum(cfg, coeffs)
dominant_scale_l = argmax(power[2:end]) + 1  # Skip l=0
characteristic_length = 40075.0 / dominant_scale_l  # km (Earth circumference)

println("Dominant spatial scale: $(characteristic_length) km")

# Global mean temperature
global_mean = real(coeffs[1]) / sqrt(4π)
println("Global mean temperature: $(global_mean) K")

destroy_config(cfg)
```

### 2. Vector Field Analysis (Atmospheric Winds)

```julia
using SHTnsKit

cfg = create_gauss_config(Float64, 32, 32, 66, 128)
θ_grid, φ_grid = get_theta(cfg), get_phi(cfg)

# Create realistic wind field: jet stream + wave pattern
u_wind = @. 30 * sin(2*θ_grid) + 15 * cos(3*φ_grid) * sin(θ_grid)  # Zonal
v_wind = @. 10 * cos(θ_grid) * sin(2*φ_grid)                        # Meridional

# Decompose into spheroidal (divergent) and toroidal (rotational) components
sph_coeffs = allocate_spectral(cfg)
tor_coeffs = allocate_spectral(cfg)
spat_to_sphtor!(cfg, u_wind, v_wind, sph_coeffs, tor_coeffs)

# Analyze flow characteristics
divergent_energy = sum(abs2, sph_coeffs)
rotational_energy = sum(abs2, tor_coeffs)
total_energy = divergent_energy + rotational_energy

println("Flow decomposition:")
println("  Divergent flow: $(100*divergent_energy/total_energy:.1f)%") 
println("  Rotational flow: $(100*rotational_energy/total_energy:.1f)%")

# Atmospheric flows are typically dominated by rotation
destroy_config(cfg)
```

### 3. Parallel Performance Analysis

```julia
using SHTnsKit

# Problem size scaling analysis
for lmax in [16, 32, 48, 64]
    cfg = create_gauss_config(Float64, lmax, lmax, 2*lmax+2, 4*lmax+1)
    
    # Get performance recommendations
    optimal_procs = optimal_process_count(cfg)
    perf_model = parallel_performance_model(cfg, optimal_procs)
    
    println("lmax=$lmax ($(cfg.nlm) coefficients):")
    println("  Recommended processes: $optimal_procs")
    println("  Expected speedup: $(perf_model.speedup:.1f)x")
    println("  Parallel efficiency: $(perf_model.efficiency*100:.1f)%")
    
    destroy_config(cfg)
end
```

### 4. Automatic Differentiation

```julia
using SHTnsKit, Zygote

cfg = create_gauss_config(Float64, 16, 16, 34, 64)

# Define optimization objective
function reconstruction_loss(sh_coeffs, target_field)
    spatial_field = allocate_spatial(cfg)
    sh_to_spat!(cfg, sh_coeffs, spatial_field)
    return sum((spatial_field - target_field).^2)
end

# Create target and initial guess
target = rand(get_nlat(cfg), get_nphi(cfg))
sh_coeffs = 0.1 * randn(ComplexF64, cfg.nlm)

# Gradient-based optimization
learning_rate = 0.001
for i in 1:100
    loss_val, grads = Zygote.withgradient(
        sh -> reconstruction_loss(sh, target), sh_coeffs)
    
    sh_coeffs .-= learning_rate .* grads[1]
    
    if i % 20 == 0
        println("Iteration $i: Loss = $loss_val")
    end
end

destroy_config(cfg)
```

##  Performance Optimization

### Threading Configuration

```julia
using SHTnsKit

# Automatic optimal threading setup
set_optimal_threads!()

# Manual control
set_threading!(true)        # Enable Julia thread parallelization  
set_fft_threads(8)         # Set FFTW thread count

# Check current settings
println("Julia threads: $(get_threading())")
println("FFTW threads: $(get_fft_threads())")
```

### Environment Variables

Control threading at startup:

```bash
export JULIA_NUM_THREADS=8
export SHTNSKIT_THREADS=on
export SHTNSKIT_FFT_THREADS=8  
export SHTNSKIT_AUTO_THREADS=on

julia --project=.
```

### Benchmarking Tools

```julia
using SHTnsKit

# Built-in comprehensive benchmark suite
run_comprehensive_benchmark(
    lmax_small=16, lmax_medium=32, lmax_large=64,
    include_scaling=true, include_precision=true, include_threading=true
)

# Memory scaling analysis
scaling_results = benchmark_memory_scaling([16, 24, 32, 40, 48])

# Vector transform performance
vector_results = benchmark_vector_transforms(cfg, n_samples=50)
```

## Parallel Computing Guide

### Running Examples

```bash
# Parallel scalar roundtrip (2 processes)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl

# Include vector field roundtrip
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --vector

# Ensure required optional packages are available (first time)
julia --project=. -e 'using Pkg; Pkg.add(["MPI","PencilArrays","PencilFFTs"])'
```

### Architecture Overview

SHTnsKit.jl uses Julia's modern package extension system:

```
SHTnsKit/
├── src/                          # Core functionality (always available)
├── ext/
│   ├── SHTnsKitParallelExt.jl   # MPI + PencilArrays + PencilFFTs
│   └── SHTnsKitLoopVecExt.jl    # LoopVectorization optimizations
└── examples/
    └── parallel_roundtrip.jl    # Distributed scalar/vector roundtrip demo
```

**Graceful Feature Detection:**
- **No optional packages**: All functions work in serial mode
- **MPI available**: Parallel computing automatically enabled
- **Full stack**: Maximum performance with all optimizations


### Automatic Differentiation

Full support for gradient-based optimization:

```julia
using SHTnsKit, ForwardDiff, Zygote

# Forward-mode differentiation
cfg = create_gauss_config(Float64, 12, 12, 26, 48)
objective(sh) = sum(abs2, sh_to_spat(cfg, sh))
gradient = ForwardDiff.gradient(objective, sh_coeffs)

# Reverse-mode differentiation (better for many parameters)
loss_val, grad = Zygote.withgradient(objective, sh_coeffs)
```

Supported functions include all core transforms, vector operations, spectral analysis, and differential operators.

### Matrix Operators

Efficient spectral differential operators:

```julia
# Apply Laplacian in spectral domain
apply_laplacian!(cfg, sh_coeffs, laplacian_result)

# cos(θ) multiplication operator  
apply_costheta_operator!(cfg, sh_coeffs, costheta_result)

# Custom matrix operations
matrix = create_custom_operator_matrix(cfg)
result = apply_matrix_operator(cfg, matrix, sh_coeffs)
```

## Scientific Applications

### Geophysical Field Analysis

```julia
# Earth's gravitational field analysis
cfg = create_gauss_config(Float64, 64, 64, 130, 256)

# Load/create gravitational potential data
gravity_field = load_gravity_data()  # Your data loading function

# Transform to spherical harmonics  
gravity_coeffs = allocate_spectral(cfg)
spat_to_sh!(cfg, gravity_field, gravity_coeffs)

# Extract specific harmonic coefficients (e.g., J2 oblateness)
j2_index = lmidx(cfg, 2, 0)
j2_coefficient = gravity_coeffs[j2_index]
println("Earth's J2 coefficient: $j2_coefficient")

# Compute power spectrum to analyze dominant spatial scales
power = power_spectrum(cfg, gravity_coeffs)
```

### Atmospheric Data Processing

```julia
# High-resolution atmospheric analysis
cfg = create_gauss_config(Float64, 85, 85, 172, 256)  # ~1.4° resolution

# Process wind field data
u_wind, v_wind = load_atmospheric_data()  # Your data

# Decompose into divergent and rotational components
div_coeffs, rot_coeffs = allocate_spectral(cfg), allocate_spectral(cfg)
spat_to_sphtor!(cfg, u_wind, v_wind, div_coeffs, rot_coeffs)

# Compute vorticity and divergence fields
vorticity = allocate_spatial(cfg)
divergence = allocate_spatial(cfg)
sphtor_to_spat!(cfg, rot_coeffs, div_coeffs, vorticity, divergence)
```

##  Testing and Validation

Run the comprehensive test suite:

```bash
# Basic functionality tests
julia --project=. -e "using Pkg; Pkg.test()"

# Parallel functionality tests  
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --vector

# Performance benchmarks
julia --project=. examples/benchmark_suite.jl

# Automatic differentiation tests
julia --project=. examples/ad_examples.jl
```

To include MPI-based roundtrip checks inside `Pkg.test()`, opt in via:

```bash
SHTNSKIT_RUN_MPI_TESTS=1 julia --project=. -e "using Pkg; Pkg.test()"
```

##  Contributing

Contributions are welcome! Areas of particular interest:

- **GPU Computing**: CUDA/ROCm support for massive parallelism
- **Advanced Algorithms**: Fast multipole methods, butterfly algorithms  
- **Domain-Specific Tools**: Climate analysis, astrophysics applications
- **Performance Optimization**: Architecture-specific tuning

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Citation

If you use SHTnsKit.jl in your research, please cite:
```bibtex
@article{schaeffer2013efficient,
  title={Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations},
  author={Schaeffer, Nathana{\"e}l},
  journal={Geochemistry, Geophysics, Geosystems},
  volume={14},
  number={3},
  pages={751--758},
  year={2013},
  publisher={Wiley Online Library}
}
```

##  License

SHTnsKit.jl is released under the GNU General Public License v3.0 (GPL-3.0), ensuring compatibility with the underlying SHTns library and promoting open scientific computing.

## References

- **[SHTns Documentation](https://nschaeff.bitbucket.io/shtns/)**: Original C library
- **[Spherical Harmonics Theory](https://en.wikipedia.org/wiki/Spherical_harmonics)**: Mathematical background  
- **[Julia Parallel Computing](https://docs.julialang.org/en/v1/manual/parallel-computing/)**: Julia parallelization guide
- **[MPI.jl Documentation](https://juliaparallel.org/MPI.jl/stable/)**: MPI interface for Julia
