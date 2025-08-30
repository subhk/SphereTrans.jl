# SHTnsKit.jl

[![Build Status](https://github.com/subhk/SHTnsKit.jl/workflows/CI/badge.svg)](https://github.com/subhk/SHTnsKit.jl/actions)

<!-- Badges -->
 <p align="left">
    <a href="https://subhk.github.io/SHTnsKit.jl/stable">
        <img alt="stable docs" src="https://img.shields.io/badge/documentation-stable%20-blue">
    </a>
      <a href="https://subhk.github.io/SHTnsKit.jl/dev">
        <img alt="latest docs" src="https://img.shields.io/badge/documentation-dev%20-orange">
    </a>
</p>

<a href="https://github.com/subhk/SHTnsKit.jl/actions/workflows/mpi-examples.yml">
  <img alt="MPI Examples" src="https://github.com/subhk/SHTnsKit.jl/actions/workflows/mpi-examples.yml/badge.svg">
</a>

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
lmax = 32
nlat = lmax + 2  # Must be ≥ lmax+1 for Gauss-Legendre accuracy
cfg = create_gauss_config(lmax, nlat; mres=2*lmax+2, nlon=4*lmax+1)

# Create test data on the sphere
spatial_data = rand(cfg.nlat, cfg.nlon)

# Transform to spherical harmonic coefficients
coeffs = analysis(cfg, spatial_data)

# Transform back to spatial domain
reconstructed = synthesis(cfg, coeffs; real_output=true)

# Check accuracy
error = maximum(abs.(spatial_data - reconstructed))
println("Roundtrip error: $error")  # Should be ~1e-14

destroy_config(cfg)
```

### Parallel Computing (MPI)

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs

MPI.Init()

# Config
lmax = 32
nlat = lmax + 2
nlon = 2*lmax + 1
cfg = create_gauss_config(lmax, nlat; nlon=nlon)

# Pencil grid: dims (:θ,:φ)
P = PencilArrays.Pencil((:θ,:φ), (nlat, nlon); comm=MPI.COMM_WORLD)
fθφ = PencilArrays.zeros(P; eltype=Float64)

# Fill some data (each rank writes to its local block)
for (iθ, iφ) in zip(eachindex(axes(fθφ,1)), eachindex(axes(fθφ,2)))
    fθφ[iθ, iφ] = sin(0.2*(iθ+1)) + cos(0.1*(iφ+1))
end

# Distributed analysis and synthesis
Alm = SHTnsKit.dist_analysis(cfg, fθφ; use_rfft=true)
fθφ2 = SHTnsKit.dist_synthesis(cfg, Alm; prototype_θφ=fθφ, real_output=true, use_rfft=true)

# Vector/QST transforms (distributed)
Vt = copy(fθφ); Vp = copy(fθφ)
Slm, Tlm = SHTnsKit.dist_spat_to_SHsphtor(cfg, Vt, Vp; use_rfft=true)
Vt2, Vp2 = SHTnsKit.dist_SHsphtor_to_spat(cfg, Slm, Tlm; prototype_θφ=Vt, real_output=true, use_rfft=true)

Vr = copy(fθφ)
Q,S,T = SHTnsKit.dist_spat_to_SHqst(cfg, Vr, Vt, Vp)
Vr2, Vt2, Vp2 = SHTnsKit.dist_SHqst_to_spat(cfg, Q, S, T; prototype_θφ=Vr, real_output=true)

MPI.Finalize()
```

### High-Performance SIMD

```julia
using SHTnsKit
using LoopVectorization  # Enables turbo optimizations

cfg = create_gauss_config(48, 48; mres=98, nlon=192)
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
cfg = create_gauss_config(42, 44; mres=86, nlon=128)  # ~2.8° resolution
θ, φ = cfg.θ, cfg.φ

# Create realistic temperature field with seasonal variation
summer_pattern = @. 273.15 + 40 * cos(θ - 0.4) + 10 * sin(2*φ) * sin(θ)
coeffs = analysis(cfg, summer_pattern)

# Analyze dominant spatial scales
power = energy_scalar_l_spectrum(cfg, coeffs)
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

cfg = create_gauss_config(32, 34; mres=66, nlon=128)
θ_grid, φ_grid = cfg.θ, cfg.φ

# Create realistic wind field: jet stream + wave pattern
u_wind = @. 30 * sin(2*θ_grid) + 15 * cos(3*φ_grid) * sin(θ_grid)  # Zonal
v_wind = @. 10 * cos(θ_grid) * sin(2*φ_grid)                        # Meridional

# Decompose into spheroidal (divergent) and toroidal (rotational) components
sph_coeffs, tor_coeffs = spat_to_SHsphtor(cfg, u_wind, v_wind)

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

### 3. Performance Analysis

```julia
using SHTnsKit

# Problem size scaling analysis
for lmax in [16, 32, 48, 64]
    cfg = create_gauss_config(lmax, lmax+2; mres=2*lmax+2, nlon=4*lmax+1)
    
    # Basic configuration information
    println("lmax=$lmax ($(cfg.nlm) coefficients):")
    println("  Grid size: $(cfg.nlat) × $(cfg.nlon)")
    println("  Total spatial points: $(cfg.nspat)")
    
    # Simple timing test
    test_data = rand(cfg.nlat, cfg.nlon)
    @time coeffs = analysis(cfg, test_data)
    @time reconstructed = synthesis(cfg, coeffs)
    
    destroy_config(cfg)
end
```

### 4. Automatic Differentiation

```julia
using SHTnsKit, Zygote

cfg = create_gauss_config(16, 16; mres=34, nlon=64)

# Define optimization objective
function reconstruction_loss(sh_coeffs, target_field)
    spatial_field = synthesis(cfg, sh_coeffs; real_output=true)
    return sum((spatial_field - target_field).^2)
end

# Create target and initial guess
target = rand(cfg.nlat, cfg.nlon)
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

### Performance Tips

- use_rfft (distributed plans): Enable real-to-complex transforms in `DistAnalysisPlan` and `DistSphtorPlan` to cut (θ,k) memory and speed real-output paths. Falls back to complex FFTs if not available.
- with_spatial_scratch (distributed vector/QST): Set to `true` to keep a single complex (θ,φ) buffer inside the plan and avoid per-call allocations for iFFT when outputs are real.
- Plan reuse: Build plans once per problem size and reuse across calls to avoid planner churn and allocations.
- Tables vs on-the-fly Plm: Precompute with `enable_plm_tables!(cfg)` to reduce CPU if your grid is fixed; results are identical to on-the-fly recurrence.

## Parallel Computing Guide

### Running Examples

```bash
# Parallel scalar roundtrip (2 processes)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl

# Include vector field roundtrip
# (Use in-place plans; add spatial scratch to avoid allocs on real outputs)
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --vector

# Include 3D (Q,S,T) roundtrip
mpiexec -n 2 julia --project=. examples/parallel_roundtrip.jl --qst

# Ensure required optional packages are available (first time)
julia --project=. -e 'using Pkg; Pkg.add(["MPI","PencilArrays","PencilFFTs"])'
 
# Distributed FFT roundtrip (2 processes)
mpiexec -n 2 julia --project=. examples/parallel_fft_roundtrip.jl
```

Spectral operator demo (cosθ application in spectral space):

```bash
mpiexec -n 2 julia --project=. examples/operator_parallel.jl           # dense
mpiexec -n 2 julia --project=. examples/operator_parallel.jl --halo    # per-m Allgatherv halo

Y-rotation demo (per-l Allgatherv over m):

```bash
mpiexec -n 2 julia --project=. examples/rotate_y_parallel.jl
```

```

Enable rfft in distributed plans (when supported):

```julia
using SHTnsKit, MPI, PencilArrays, PencilFFTs
MPI.Init()
cfg = create_gauss_config(16, 18; nlon=33)
Pθφ = PencilArrays.Pencil((:θ,:φ), (cfg.nlat, cfg.nlon); comm=MPI.COMM_WORLD)

# Scalar analysis with rfft
aplan = DistAnalysisPlan(cfg, PencilArrays.zeros(Pθφ; eltype=Float64); use_rfft=true)

# Vector transforms with rfft + optional spatial scratch to avoid iFFT allocs for real outputs
vplan = DistSphtorPlan(cfg, PencilArrays.zeros(Pθφ; eltype=Float64); use_rfft=true, with_spatial_scratch=true)
MPI.Finalize()
```

### Automatic Differentiation

Full support for gradient-based optimization:

```julia
using SHTnsKit, ForwardDiff, Zygote

# Forward-mode differentiation
cfg = create_gauss_config(12, 12; mres=26, nlon=48)
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

### Allocation Benchmarks

```bash
# Serial and (if available) MPI allocation benchmarks
julia --project=. examples/alloc_benchmark.jl 16
mpiexec -n 2 julia --project=. examples/alloc_benchmark.jl 16

Tip: To avoid allocations for real-output distributed synthesis, construct plans with `with_spatial_scratch=true`, which keeps a single complex (θ,φ) scratch buffer inside the plan. This modest, fixed footprint removes per-call allocations for iFFT writes when outputs are real.
```

##  Contributing

Contributions are welcome! Areas of particular interest:

- **GPU Computing**: CUDA/ROCm support for massive parallelism
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

