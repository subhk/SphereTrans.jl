# SHTnsKit.jl Troubleshooting Guide

This guide helps you diagnose and resolve common issues with SHTnsKit.jl, particularly those related to SHTns binary compatibility.

## Table of Contents
- [Quick Diagnosis](#quick-diagnosis)
- [Common Issues](#common-issues)
- [SHTns_jll Binary Problems](#shtns_jll-binary-problems)
- [Platform-Specific Issues](#platform-specific-issues)
- [Performance Issues](#performance-issues)
- [Advanced Debugging](#advanced-debugging)

## Quick Diagnosis

First, run this diagnostic script to identify your issue:

```julia
using SHTnsKit

# Quick system check
println("Platform: $(Sys.KERNEL) $(Sys.ARCH)")
println("Julia version: $(VERSION)")

# Check SHTns availability
println("\n=== SHTns Library Status ===")
has_symbols = SHTnsKit.has_shtns_symbols()
should_test = SHTnsKit.should_test_shtns_by_default()

println("Has SHTns symbols: $has_symbols")
println("Should test by default: $should_test")
println("Library path: $(SHTnsKit.get_library_path())")

# Validation check
try
    is_valid = SHTnsKit.validate_library()
    println("Library validation: $(is_valid ? "✓ PASSED" : "✗ FAILED")")
catch e
    println("Library validation: ✗ ERROR - $e")
end

# Basic functionality test
try
    cfg = create_test_config(4, 4)
    nlm = get_nlm(cfg)
    free_config(cfg)
    println("Basic functionality: ✓ WORKING (nlm=$nlm)")
catch e
    println("Basic functionality: ✗ FAILED - $e")
end
```

## Common Issues

### 1. "undefined symbol" Errors

**Symptoms:**
```
ERROR: ccall: could not find function shtns_get_lmax
ERROR: undefined symbol: shtns_get_lmax
```

**Cause:** Missing or incomplete SHTns_jll binary package.

**Solutions:**
```julia
# Option A: Use custom SHTns installation
ENV["SHTNS_LIBRARY_PATH"] = "/path/to/your/libshtns.so"
using SHTnsKit

# Option B: Force enable testing (risky)
ENV["SHTNSKIT_TEST_SHTNS"] = "true"
using SHTnsKit

# Option C: Build from source (recommended)
# See "Building SHTns from Source" section below
```

### 2. "nlat or nphi is zero" Errors

**Symptoms:**
```
ERROR: nlat or nphi is zero in SHTns configuration
```

**Cause:** SHTns_jll binary accuracy issues with grid parameter computation.

**Solutions:**
```julia
# Use test configuration instead of standard configs
cfg = create_test_config(lmax, mmax)  # Instead of create_gauss_config

# Or use try-catch fallback pattern
try
    cfg = create_gauss_config(16, 16)
catch e
    @warn "Standard config failed, using test config: $e"
    cfg = create_test_config(16, 16)
end
```

### 3. Segmentation Faults

**Symptoms:**
```
signal (11): Segmentation fault
```

**Cause:** Array size mismatches or NULL pointer access.

**Solutions:**
```julia
# Always validate arrays before transforms
cfg = create_gauss_config(16, 16)
nlm = get_nlm(cfg)
nlat, nphi = get_nlat(cfg), get_nphi(cfg)

# Correct way
sh = Vector{Float64}(undef, nlm)        # ✓ Correct size
spat = Matrix{Float64}(undef, nlat, nphi) # ✓ Correct size

# Wrong way that causes segfaults
# sh = Vector{Float64}(undef, 10)       # ✗ Wrong size
# spat = Matrix{Float64}(undef, 5, 5)   # ✗ Wrong size
```

### 4. Performance Issues

**Symptoms:** Slow transforms, high memory usage.

**Solutions:**
```julia
# Set optimal threading
nthreads = set_optimal_threads()
println("Using $nthreads threads")

# Use in-place operations
synthesize!(cfg, sh, spat)  # Instead of synthesize(cfg, sh)

# Batch operations on same config
for i in 1:n_transforms
    synthesize!(cfg, sh_data[i], spat_data[i])
end
# Better than creating new configs each time
```

## SHTns_jll Binary Problems

### Understanding the Issue

SHTns_jll (the binary package for SHTns) has known compatibility issues:

1. **Missing symbols**: Some builds lack newer SHTns functions
2. **Accuracy problems**: Grid parameter computation can fail
3. **Platform issues**: Different behavior on different OS/architectures

### Checking Your SHTns_jll Status

```julia
using SHTnsKit
import Pkg

# Check SHTns_jll version
try
    import SHTns_jll
    println("SHTns_jll version: $(SHTns_jll.version)")
    println("SHTns_jll libshtns path: $(SHTns_jll.libshtns_path)")
catch e
    println("SHTns_jll not available: $e")
end

# Check for specific symbols
symbols_to_check = [
    :shtns_get_lmax, :shtns_get_mmax, :shtns_get_nlat, :shtns_get_nphi,
    :shtns_SH_to_point, :shtns_SH_to_grad_spat, :shtns_SHqst_to_point
]

for sym in symbols_to_check
    try
        SHTnsKit.require_symbol(sym)
        println("✓ $sym available")
    catch
        println("✗ $sym missing")
    end
end
```

### Building SHTns from Source

For the most reliable experience, build SHTns from source:

```bash
# Download and build SHTns
wget https://bitbucket.org/nschaeff/shtns/downloads/shtns-3.7.0.tar.gz
tar -xzf shtns-3.7.0.tar.gz
cd shtns-3.7.0

# Configure with optimal settings
./configure \
    --enable-openmp \
    --enable-ishioka \
    --enable-magic-layout \
    --enable-gpu \
    --prefix=/usr/local

make -j$(nproc)
sudo make install
```

Then use it in Julia:
```julia
ENV["SHTNS_LIBRARY_PATH"] = "/usr/local/lib/libshtns.so"
using SHTnsKit
```

## Platform-Specific Issues

### macOS

**Issue:** SHTns_jll binaries often have symbol issues on macOS.

**Solutions:**
```bash
# Install via Homebrew (if available)
brew install shtns

# Or build from source with specific flags
./configure --enable-openmp --with-libpthread
```

```julia
# Point to Homebrew installation
ENV["SHTNS_LIBRARY_PATH"] = "/opt/homebrew/lib/libshtns.dylib"
using SHTnsKit
```

### Linux

**Issue:** Missing OpenMP or FFTW dependencies.

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get install libfftw3-dev libomp-dev

# CentOS/RHEL
sudo yum install fftw-devel libgomp-devel

# Then build SHTns from source
```

### Windows

**Issue:** Limited SHTns support on Windows.

**Solutions:**
```julia
# Use Windows Subsystem for Linux (WSL)
# Or Docker container with Linux environment

# Alternative: Use fallback mode
ENV["SHTNSKIT_TEST_SHTNS"] = "false"  # Disable SHTns-dependent tests
using SHTnsKit
# Only basic functionality will work
```

## Performance Issues

### Slow Transforms

**Diagnosis:**
```julia
using BenchmarkTools

cfg = create_gauss_config(64, 64)
sh = rand(get_nlm(cfg))

# Benchmark synthesis
@benchmark synthesize($cfg, $sh)
```

**Optimizations:**
```julia
# 1. Set optimal thread count
set_optimal_threads()

# 2. Use in-place operations
spat = allocate_spatial(cfg)
@benchmark synthesize!($cfg, $sh, $spat)

# 3. Choose optimal grid type
cfg_gauss = create_gauss_config(64, 64)    # Usually faster
cfg_regular = create_regular_config(64, 64) # Sometimes better for specific cases

# 4. Batch operations
configs = [create_gauss_config(32, 32) for _ in 1:4]
@time for cfg in configs
    spat = synthesize(cfg, sh)
end
```

### Memory Issues

**Diagnosis:**
```julia
using Profile

# Profile memory allocations
@profile begin
    for i in 1:100
        cfg = create_gauss_config(32, 32)
        spat = synthesize(cfg, rand(get_nlm(cfg)))
        free_config(cfg)
    end
end

Profile.print(format=:flat, sortedby=:count)
```

**Solutions:**
```julia
# Reuse configurations
cfg = create_gauss_config(32, 32)
sh = allocate_spectral(cfg)
spat = allocate_spatial(cfg)

for i in 1:1000
    # Reuse arrays instead of allocating new ones
    rand!(sh)
    synthesize!(cfg, sh, spat)
    # Process spat...
end

free_config(cfg)
```

## Advanced Debugging

### Enable Debug Logging

```julia
using Logging

# Enable debug messages
global_logger(ConsoleLogger(stderr, Logging.Debug))

# Now SHTnsKit will show debug info about fallbacks
cfg = create_gauss_config(16, 16)  # May show debug messages about missing symbols
```

### Symbol Availability Check

```julia
using Libdl

function check_shtns_symbols()
    lib_path = SHTnsKit.get_library_path()
    println("Checking symbols in: $lib_path")
    
    try
        handle = dlopen(lib_path, RTLD_LAZY)
        
        symbols = [
            "shtns_create", "shtns_sh_to_spat", "shtns_spat_to_sh",
            "shtns_get_lmax", "shtns_get_mmax", "shtns_get_nlat",
            "shtns_SH_to_point", "shtns_SHqst_to_spat", "shtns_spat_to_SHqst"
        ]
        
        for sym in symbols
            has_sym = dlsym_e(handle, sym) != C_NULL
            println("  $(has_sym ? "✓" : "✗") $sym")
        end
        
        dlclose(handle)
    catch e
        println("Error checking symbols: $e")
    end
end

check_shtns_symbols()
```

### Memory Debugging

```julia
# Check for memory leaks in config creation/destruction
function test_memory_leak()
    initial_memory = Base.gc_total_bytes(Base.gc_num())
    
    for i in 1:1000
        cfg = create_gauss_config(16, 16)
        free_config(cfg)
        
        if i % 100 == 0
            GC.gc()
            current_memory = Base.gc_total_bytes(Base.gc_num())
            println("Iteration $i: $(current_memory - initial_memory) bytes")
        end
    end
end

test_memory_leak()
```

## Getting Help

If you're still experiencing issues:

1. **Check the issues**: Visit [SHTnsKit.jl Issues](https://github.com/subhk/SHTnsKit.jl/issues)

2. **Create a minimal example**:
```julia
using SHTnsKit

# Include your problematic code here
# Make it as minimal as possible
```

3. **Provide system information**:
```julia
using InteractiveUtils
versioninfo()

# Also include the diagnostic script output from the beginning of this guide
```

4. **Consider alternative workflows**:
```julia
# Use test configurations instead of standard ones
cfg = create_test_config(lmax, mmax)

# Use manual memory management
spat = zeros(get_nlat(cfg), get_nphi(cfg))
synthesize!(cfg, sh, reshape(spat, :))

# Implement custom fallbacks for missing functionality
```

## Summary

Most SHTnsKit.jl issues stem from SHTns_jll binary compatibility problems. The recommended solutions are:

1. **Build SHTns from source** for best reliability
2. **Use test configurations** when standard configs fail  
3. **Enable comprehensive validation** to catch errors early
4. **Optimize for your platform** with proper threading and memory management

SHTnsKit.jl is designed to be robust against these issues with extensive fallback mechanisms and validation.