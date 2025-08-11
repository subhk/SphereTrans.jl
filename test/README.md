# SHTnsKit.jl Test Suite

This directory contains a comprehensive test suite for SHTnsKit.jl that covers all functionality of the SHTns C library wrapper.

## Test Structure

### Main Test Runner
- **`runtests.jl`** - Main test runner that includes all test modules

### Core Functionality Tests
- **`test_basic.jl`** - Basic SHTns functionality tests:
  - Configuration creation and grid setup
  - Memory allocation and management
  - Grid coordinates and Gauss weights
  - Transform accuracy with known spherical harmonics
  - Error handling for invalid inputs

### Advanced Feature Tests
- **`test_vector.jl`** - Vector field transform tests:
  - Spheroidal-toroidal decomposition accuracy
  - Gradient and curl operations
  - Vector energy conservation
  - Helmholtz decomposition verification
  - In-place vector operations

- **`test_complex.jl`** - Complex field transform tests:
  - Complex spectral-spatial transforms
  - Complex vs real transform consistency
  - Complex field properties (pure imaginary, etc.)
  - Transform linearity verification
  - Memory management for complex types

- **`test_rotation.jl`** - Field rotation and advanced features:
  - Spectral and spatial field rotations
  - Power spectrum analysis
  - Rotation composition and identity tests
  - Energy/norm conservation during rotations

### Performance and Parallelization Tests
- **`test_threading.jl`** - Threading and parallelization:
  - OpenMP thread control functions
  - Thread safety for concurrent transforms
  - Multi-configuration concurrent operations
  - Threading performance analysis
  - Thread-safe error handling

- **`test_gpu.jl`** - GPU acceleration tests:
  - GPU initialization and cleanup
  - GPU vs CPU transform accuracy
  - GPU memory management
  - GPU performance benchmarking
  - Error handling on GPU

- **`test_mpi.jl`** - MPI distributed computing tests:
  - MPI extension loading and availability
  - MPI symbol detection mechanisms
  - MPI configuration type handling
  - Environment variable processing
  - Fallback behavior when MPI symbols unavailable

### Performance Standards
- **`test_benchmarks.jl`** - Performance benchmarks and standards:
  - Transform performance time limits
  - Memory allocation efficiency
  - Scalability with problem size
  - Threading performance impact
  - Numerical accuracy standards
  - Stress testing with repeated operations

## Running Tests

### Prerequisites
- SHTns C library installed and available
- Julia environment with SHTnsKit.jl dependencies

### Run All Tests
```bash
julia --project=. test/runtests.jl
```

### Run Individual Test Modules
```bash
julia --project=. -e "using Test; include(\"test/test_basic.jl\")"
julia --project=. -e "using Test; include(\"test/test_vector.jl\")"
julia --project=. -e "using Test; include(\"test/test_gpu.jl\")"
# etc.
```

### Test Structure Verification
```bash
julia test_simple.jl  # Verify syntax without running tests
```

## Test Coverage

The test suite covers:

✅ **Core SHTns Features:**
- Forward/backward scalar transforms
- Multiple grid types (Gauss, regular)
- Configuration management
- Memory allocation

✅ **Advanced Features:**
- Complex field transforms
- Vector field transforms (spheroidal-toroidal)
- Field rotations (Wigner D-matrices)
- Power spectrum analysis
- Gradient and curl operations

✅ **Performance Optimizations:**
- OpenMP multi-threading
- GPU acceleration (CUDA)
- Memory management efficiency
- Numerical accuracy standards

✅ **Distributed Computing:**
- MPI support and fallbacks
- Thread safety
- Error handling

✅ **Quality Assurance:**
- Performance benchmarks
- Stress testing
- Edge case handling
- Memory leak detection

## Test Philosophy

- **Robustness:** Tests gracefully handle missing dependencies (SHTns library, CUDA, MPI)
- **Accuracy:** Strict numerical accuracy requirements (< 1e-12 for round-trip transforms)
- **Performance:** Reasonable time and memory limits for CI/CD environments
- **Completeness:** Cover all exported functions and edge cases
- **Maintainability:** Modular test structure for easy extension

## Adding New Tests

When adding new functionality to SHTnsKit.jl:

1. Add tests to appropriate test module (or create new one)
2. Update `runtests.jl` to include new test file
3. Follow existing patterns for error handling and skip conditions
4. Ensure tests work without optional dependencies (GPU, MPI)
5. Add performance benchmarks for computationally expensive features

## Continuous Integration

The test suite is designed to work in CI environments:
- Graceful handling of missing system dependencies
- Reasonable time limits for test completion
- Memory usage appropriate for standard CI runners
- Skip patterns for unavailable optional features