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
- **Memory**: At least 4GB RAM recommended
- **Storage**: 500MB free space for dependencies

### Required Dependencies

SHTnsKit.jl is pure Julia and does not require an external C library. The only runtime dependencies are Julia’s standard libraries and FFTW.jl (installed automatically).

## Installing SHTnsKit.jl

### Standard Installation

```julia
using Pkg
Pkg.add("SHTnsKit")
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

## Optional Dependencies

### Additional Performance Utilities

```julia
using Pkg
Pkg.add(["FFTW", "LinearAlgebra"])
```

These are automatically included but explicit installation ensures optimal versions.

## Verification

### Basic Functionality Test

```julia
using SHTnsKit

# Create simple configuration
cfg = create_gauss_config(8, 8)
println("lmax: ", get_lmax(cfg))
println("nlat: ", get_nlat(cfg))  
println("nphi: ", get_nphi(cfg))

# Test basic transform
sh = rand(get_nlm(cfg))
spat = synthesize(cfg, sh)
println("Transform successful: ", size(spat))

destroy_config(cfg)
println(" SHTnsKit.jl installation verified!")
```

### Extended Verification

```julia
using SHTnsKit, Test

@testset "Installation Verification" begin
    # Basic functionality
    cfg = create_gauss_config(16, 16)
    sh = rand(get_nlm(cfg))
    spat = synthesize(cfg, sh)
    sh2 = analyze(cfg, spat)
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

**Fix:** Ensure `length(sh) == get_nlm(cfg)` and `size(spatial) == (get_nlat(cfg), get_nphi(cfg))`.

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
