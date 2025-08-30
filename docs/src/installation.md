# Installation Guide

This guide provides detailed instructions for installing SHTnsKit.jl and its dependencies.

## Quick Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
```

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows with WSL
- **Julia**: Version 1.9 or later (1.11+ recommended)
- **Memory**: At least 4GB RAM (16GB+ for large parallel problems)
- **Storage**: 2GB free space for dependencies (including MPI)
- **MPI Library**: OpenMPI or MPICH for parallel functionality

### Required Dependencies

SHTnsKit.jl is pure Julia and does not require an external C library. Core functionality uses Julia's standard libraries and FFTW.jl (installed automatically). Parallel features require additional packages.

## Installing SHTnsKit.jl

### Basic Installation (Serial Only)

```julia
using Pkg
Pkg.add("SHTnsKit")
```

### Full Installation (Parallel + SIMD)

```julia
using Pkg
Pkg.add(["SHTnsKit", "MPI", "PencilArrays", "PencilFFTs", "LoopVectorization"])
```

### Development Installation

For the latest features or contributing:

```julia
using Pkg
Pkg.add(url="https://github.com/username/SHTnsKit.jl.git")
```

### Local Development Setup

```julia
using Pkg
Pkg.develop(path="/path/to/SHTnsKit.jl")
```

## Parallel Computing Setup

### MPI Installation

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libopenmpi-dev openmpi-bin
```

**macOS:**
```bash
brew install open-mpi
```

**Configure Julia MPI:**
```julia
using Pkg
Pkg.add("MPI")
Pkg.build("MPI")
```

### Verify MPI Installation

```julia
using MPI
MPI.Init()
rank = Comm_rank(COMM_WORLD)
size = Comm_size(COMM_WORLD)
println("Process $rank of $size")
MPI.Finalize()
```

### Optional Performance Packages

```julia
using Pkg
Pkg.add(["LoopVectorization", "BenchmarkTools"])
```

## Verification

### Basic Functionality Test

```julia
using SHTnsKit

# Create simple configuration
cfg = create_gauss_config(8, 8)
println("lmax: ", get_lmax(cfg))
println("nlat: ", cfg.nlat)  
println("nphi: ", cfg.nlon)

# Test basic transform
# Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
spat = synthesis(cfg, sh)
println("Transform successful: ", size(spat))

destroy_config(cfg)
println("SHTnsKit.jl installation verified!")
```

### Parallel Functionality Test

```julia
# Test serial mode (no MPI required)
using SHTnsKit

cfg = create_gauss_config(10, 8; mres=24, nlon=32)
sh_coeffs = randn(Complex{Float64}, cfg.nlm)

# This should work without MPI packages
try
    auto_cfg = auto_parallel_config(cfg)
    println("Serial fallback working")
catch
    println("Parallel packages not detected (expected)")
end

# Test with MPI packages (run with: mpiexec -n 2 julia script.jl)
try
    using MPI, PencilArrays, PencilFFTs
    MPI.Init()
    
    pcfg = create_parallel_config(cfg, MPI.COMM_WORLD)
    result = similar(sh_coeffs)
    parallel_apply_operator(pcfg, :laplacian, sh_coeffs, result)
    
    println("Parallel functionality verified!")
    MPI.Finalize()
catch e
    println("INFO: Parallel packages not available: $e")
end
```

### Extended Verification

```julia
using SHTnsKit, Test

@testset "Installation Verification" begin
    # Basic functionality
    cfg = create_gauss_config(16, 16)
    # Create bandlimited test coefficients (avoids high-frequency errors)
sh = zeros(cfg.nlm)
sh[1] = 1.0
if cfg.nlm > 3
    sh[3] = 0.5
end
    spat = synthesis(cfg, sh)
    sh2 = analysis(cfg, spat)
    @test norm(sh - sh2) < 1e-12
    
    # Threading (FFTW thread setting available)
    @test get_fft_threads() >= 1
    
    # Memory management
    destroy_config(cfg)
    @test true  # No crash
end
```

## Troubleshooting

### Common Issues

**1. Array size mismatch:**
```
ERROR: DimensionMismatch: spatial_data size (X, Y) must be (nlat, nphi)
```

**Fix:** Ensure `length(sh) == cfg.nlm` and `size(spatial) == (cfg.nlat, cfg.nlon)`.

**2. Memory issues:**
```
ERROR: Out of memory
```

**Solutions:**
- Reduce problem size (lmax, mmax)
- Increase system swap space
 - Reuse allocations with in‑place APIs (`synthesize!`, `analyze!`)

### Advanced Debugging

**Julia environment check:**
```julia
using Libdl
println(Libdl.dllist())  # List all loaded libraries
```

## Performance Optimization

### System-Level Optimizations

Threading and memory tips:
```julia
# Enable SHTnsKit internal threading and FFTW threads
set_optimal_threads!()
println((threads=get_threading(), fft_threads=get_fft_threads()))

# Prevent oversubscription with BLAS/FFTW (optional)
ENV["OPENBLAS_NUM_THREADS"] = "1"
```

### Julia-Specific

**Precompilation:**
```julia
using PackageCompiler
create_sysimage([:SHTnsKit]; sysimage_path="shtns_sysimage.so")
```

**Memory:**
```bash
julia --heap-size-hint=8G script.jl
```

## Docker Installation

For containerized environments:

```dockerfile
FROM julia:1.11

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.add(["SHTnsKit"])'

# Verify installation
RUN julia -e 'using SHTnsKit; cfg = create_gauss_config(8,8); destroy_config(cfg)'
```

## Getting Help

- **Documentation**: [SHTnsKit.jl Docs](https://subhk.github.io/SHTnsKit.jl/)
- **Issues**: [GitHub Issues](https://github.com/subhk/SHTnsKit.jl/issues)
- **Julia Discourse**: [Julia Community](https://discourse.julialang.org/)
