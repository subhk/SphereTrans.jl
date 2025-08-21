"""
    SHTnsKit.jl

A pure Julia implementation of spherical harmonic transforms (SHT) for numerical simulations.
This is a native Julia translation of the SHTns C library, providing fast and efficient 
spherical harmonic transforms without external C dependencies.

Key Features:
- Pure Julia implementation of spherical harmonic transforms
- Scalar and vector transforms
- Multiple grid types (Gauss-Legendre, regular)
- High performance with SIMD optimizations
- Thread-safe operations
- Memory-efficient algorithms
"""
module SHTnsKit

using LinearAlgebra
using FFTW
using Base.Threads

export 
    # Core types
    SHTnsConfig, SHTnsNorm, SHTnsType, SHTnsGrid,
    
    # Configuration and setup
    create_config, set_grid!, destroy_config,
    
    # Grid utilities
    get_lmax, get_mmax, get_nlat, get_nphi, get_nlm,
    get_theta, get_phi, get_gauss_weights,
    
    # Core transforms
    sh_to_spat!, spat_to_sh!,
    sh_to_spat, spat_to_sh,
    
    # Vector transforms
    sphtor_to_spat!, spat_to_sphtor!,
    
    # Complex transforms
    cplx_sh_to_spat!, cplx_spat_to_sh!,
    
    # Utility functions
    lmidx, nlm_calc, allocate_spectral, allocate_spatial,
    
    # Grid creation helpers
    create_gauss_config, create_regular_config

include("types.jl")
include("gauss_legendre.jl") 
include("fft_utils.jl")
include("core_transforms.jl")
include("vector_transforms.jl")
include("complex_transforms.jl")
include("utilities.jl")
include("grid_utils.jl")

end # module SHTnsKit