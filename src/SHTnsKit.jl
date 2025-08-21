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
    lmidx, lm_from_index, nlm_calc, allocate_spectral, allocate_spatial,
    synthesize, analyze, synthesize!, analyze!,
    real_nlm, analyze_real, synthesize_real, complex_to_real_coeffs, real_to_complex_coeffs,
    
    # Analysis functions
    evaluate_at_point, power_spectrum, total_power,
    spatial_integral, spatial_mean,
    spatial_divergence, spatial_vorticity,
    
    # Vector transforms
    synthesize_vector, analyze_vector,
    analyze_vector_real, synthesize_vector_real,
    
    # Complex transforms  
    synthesize_complex, analyze_complex,
    
    # Complex spectral ops
    cplx_spectral_derivative_phi, cplx_spectral_laplacian, cplx_spatial_derivatives,
    cplx_spectral_gradient_spatial, cplx_divergence_spectral, cplx_vorticity_spectral,
    cplx_vector_from_spheroidal, cplx_vector_from_toroidal, cplx_vector_from_potentials,
    cplx_divergence_spatial_from_potentials, cplx_vorticity_spatial_from_potentials,
    cplx_sphtor_to_spat!, cplx_synthesize_vector, cplx_spat_to_sphtor!, cplx_analyze_vector,
    rotate_complex!, rotate_real!,
    
    # Internal functions (for testing)
    compute_gauss_legendre_nodes_weights, compute_associated_legendre,
    
    # Grid creation helpers
    create_gauss_config, create_regular_config,
    
    # Threading controls
    set_threading!, get_threading, set_fft_threads, get_fft_threads, set_optimal_threads!

include("types.jl")
include("gauss_legendre.jl") 
include("fft_utils.jl")
include("core_transforms.jl")
include("vector_transforms.jl")
include("complex_transforms.jl")
include("utilities.jl")
include("grid_utils.jl")
include("threading.jl")


function __init__()
    # Optional environment-driven defaults
    if haskey(ENV, "SHTNSKIT_THREADS")
        val = lowercase(String(ENV["SHTNSKIT_THREADS"]))
        flag = val in ("1", "true", "yes", "on")
        try
            set_threading!(flag)
        catch
        end
    end
    if haskey(ENV, "SHTNSKIT_FFT_THREADS")
        s = String(ENV["SHTNSKIT_FFT_THREADS"]) 
        try
            set_fft_threads(parse(Int, s))
        catch
        end
    end
    if haskey(ENV, "SHTNSKIT_AUTO_THREADS")
        val = lowercase(String(ENV["SHTNSKIT_AUTO_THREADS"]))
        if val in ("1", "true", "yes", "on")
            try
                set_optimal_threads!()
            catch
            end
        end
    end
end


end # module SHTnsKit
