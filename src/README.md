# SHTnsKit.jl Source Code Structure

This directory contains the organized source code for SHTnsKit.jl, structured for easy maintenance and debugging.

## 📁 Directory Structure

### `core/` - Core Functionality
- **`types_optimized.jl`** - Core data types and configurations
- **`gauss_legendre.jl`** - Gauss-Legendre quadrature implementation
- **`fft_utils.jl`** - FFT utilities and wrappers
- **`core_transforms.jl`** - Basic spherical harmonic transforms

### `transforms/` - Transform Algorithms
- **`vector_transforms.jl`** - Vector spherical harmonic transforms
- **`complex_transforms.jl`** - Complex number transforms
- **`fast_transforms.jl`** - Fast transform algorithms (O(L² log L))
- **`matrix_operators.jl`** - Matrix operators (Laplacian, cos θ, etc.)
- **`single_m_transforms.jl`** - Single azimuthal mode transforms
- **`truncated_transforms.jl`** - Truncated/partial transforms

### `advanced/` - Advanced Optimizations
- **`advanced_hybrid_algorithms.jl`** - Adaptive algorithm selection
- **`advanced_parallel_transforms.jl`** - Multi-level parallelism (MPI+OpenMP+SIMD)
- **`advanced_communication_patterns.jl`** - Topology-aware communication
- **`advanced_memory_optimization.jl`** - Cache-aware and NUMA-optimized memory management
- **`advanced_performance_tuning.jl`** - Machine learning-based auto-tuning

### `utils/` - Utilities and Helpers
- **`utilities.jl`** - General utility functions
- **`grid_utils.jl`** - Grid generation and management
- **`threading.jl`** - Threading configuration and controls
- **`point_evaluation.jl`** - Point evaluation functions
- **`special_functions.jl`** - Special mathematical functions
- **`robert_form.jl`** - Robert form transforms

### `benchmarks/` - Performance Testing
- **`profiling.jl`** - Profiling and timing utilities
- **`benchmarking_suite.jl`** - Comprehensive benchmarking framework

## 🔧 Module Loading Order

The main `SHTnsKit.jl` file loads components in logical dependency order:

1. **Core functionality** - Types and basic operations
2. **Transform algorithms** - All transform implementations
3. **Utilities** - Helper functions and configuration
4. **Benchmarking** - Performance testing tools
5. **Advanced functionality** - Loaded automatically via extensions when needed

## 🚀 Advanced Features

Advanced optimizations are implemented as separate modules that can be used independently or in combination:

- **Hybrid algorithms** automatically select the best implementation
- **Multi-level parallelism** scales from single-core to HPC clusters
- **Communication optimization** adapts to network topology
- **Memory optimization** leverages cache hierarchy and NUMA
- **Auto-tuning** learns optimal parameters for each system

## 📚 Usage

```julia
using SHTnsKit

# Core functionality is always available
cfg = create_gauss_config(Float64, 128, 128)
sh_coeffs = randn(cfg.nlm)
spatial_data = Matrix{Float64}(undef, cfg.nlat, cfg.nphi)

# Basic transforms
sh_to_spat!(cfg, sh_coeffs, spatial_data)

# Advanced features activate automatically when dependencies are available
# No code changes needed - just install MPI, PencilArrays, etc.
```

This organization makes the codebase much easier to navigate, debug, and maintain while preserving all functionality.