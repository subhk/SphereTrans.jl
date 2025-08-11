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

SHTnsKit.jl requires the **SHTns C library** to be installed on your system. This is the most critical dependency.

## Installing SHTns C Library

### Option 1: Package Manager (Recommended)

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install libshtns-dev
```

**macOS (Homebrew):**
```bash
brew install shtns
```

**Arch Linux:**
```bash
yay -S shtns
```

### Option 2: Build from Source

If package managers don't work or you need the latest version:

```bash
# Download and extract SHTns
wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-3.7.tar.gz
tar -xzf shtns-3.7.tar.gz
cd shtns-3.7

# Configure and build
./configure --enable-openmp --enable-python
make
sudo make install

# Update library path (Linux)
sudo ldconfig
```

**Configuration Options:**
- `--enable-openmp`: Enable OpenMP multi-threading
- `--enable-python`: Build Python interface (optional)
- `--enable-cuda`: Enable CUDA GPU support (if NVIDIA GPU available)
- `--enable-ishioka`: Enable Ishioka optimization for high-degree transforms

### Verify SHTns Installation

```bash
# Check if library is found
ldconfig -p | grep shtns

# Or check specific location
ls -la /usr/local/lib/libshtns*
```

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

### GPU Support (CUDA)

For GPU acceleration:

```julia
using Pkg
Pkg.add("CUDA")
```

**System Requirements:**
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 11.0+ installed
- cuFFT library available

**Verify CUDA Setup:**
```julia
using CUDA
CUDA.functional()  # Should return true
```

### MPI Distributed Computing

For multi-node parallelization:

```julia
using Pkg
Pkg.add("MPI")
```

**System Requirements:**
- MPI implementation (OpenMPI, MPICH, Intel MPI)
- Configured MPI environment

**Verify MPI Setup:**
```bash
mpirun --version
which mpirun
```

### Additional Performance Dependencies

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

free_config(cfg)
println("✓ SHTnsKit.jl installation verified!")
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
    
    # Threading
    @test get_num_threads() >= 1
    
    # Memory management
    free_config(cfg)
    @test true  # No crash
end
```

## Troubleshooting

### Common Issues

**1. SHTns library not found:**
```
ERROR: could not load library "libshtns"
```

**Solutions:**
- Verify SHTns installation: `ldconfig -p | grep shtns`
- Set `LD_LIBRARY_PATH`: `export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH`
- Rebuild Julia packages: `Pkg.build("SHTnsKit")`

**2. OpenMP issues:**
```
WARNING: OpenMP not available
```

**Solutions:**
- Install OpenMP: `sudo apt install libomp-dev` (Ubuntu)
- Rebuild SHTns with `--enable-openmp`
- Set thread count: `export OMP_NUM_THREADS=4`

**3. CUDA not functional:**
```
ERROR: CUDA not functional
```

**Solutions:**
- Check NVIDIA drivers: `nvidia-smi`
- Verify CUDA installation: `nvcc --version`
- Rebuild CUDA.jl: `Pkg.build("CUDA")`

**4. Memory issues:**
```
ERROR: Out of memory
```

**Solutions:**
- Reduce problem size (lmax, mmax)
- Increase system swap space
- Use GPU offloading for large problems

### Advanced Debugging

**Check library symbols:**
```bash
nm -D /usr/local/lib/libshtns.so | grep shtns_
```

**Test SHTns directly:**
```c
// test_shtns.c
#include <shtns.h>
int main() {
    printf("SHTns version: %s\n", shtns_version());
    return 0;
}
```

```bash
gcc test_shtns.c -lshtns -lm -o test_shtns
./test_shtns
```

**Julia environment check:**
```julia
using Libdl
println(Libdl.dllist())  # List all loaded libraries
```

## Performance Optimization

### System-Level Optimizations

**CPU Affinity:**
```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

**Memory:**
```bash
export OMP_NUM_THREADS=4  # Match physical cores
export OPENBLAS_NUM_THREADS=1  # Avoid oversubscription
```

**NUMA:**
```bash
numactl --interleave=all julia script.jl
```

### Julia-Specific

**Precompilation:**
```julia
using PackageCompiler
create_sysimage([:SHTnsKit, :CUDA, :MPI]; sysimage_path="shtns_sysimage.so")
```

**Memory:**
```bash
julia --heap-size-hint=8G script.jl
```

## Docker Installation

For containerized environments:

```dockerfile
FROM julia:1.11

# Install SHTns dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libfftw3-dev \
    libshtns-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Julia packages
RUN julia -e 'using Pkg; Pkg.add(["SHTnsKit", "CUDA", "MPI"])'

# Verify installation
RUN julia -e 'using SHTnsKit; cfg = create_gauss_config(8,8); free_config(cfg)'
```

## Getting Help

- **Documentation**: [SHTnsKit.jl Docs](https://username.github.io/SHTnsKit.jl/)
- **Issues**: [GitHub Issues](https://github.com/username/SHTnsKit.jl/issues)  
- **SHTns Documentation**: [SHTns Manual](https://nschaeff.bitbucket.io/shtns/)
- **Julia Discourse**: [Julia Community](https://discourse.julialang.org/)