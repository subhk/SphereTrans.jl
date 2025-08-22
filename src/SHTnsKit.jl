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
    set_threading!, get_threading, set_fft_threads, get_fft_threads, set_optimal_threads!,
    
    # Point evaluation functions
    sh_to_point, sh_to_point_cplx, shqst_to_point, sh_to_grad_point,
    
    # Special value functions
    sh00_1, sh10_ct, sh11_st, shlm_e1, gauss_weights,
    
    # Legendre polynomial evaluation
    legendre_sphPlm_array, legendre_sphPlm_deriv_array,
    
    # Matrix operators
    mul_ct_matrix, st_dt_matrix, sh_mul_mx, 
    apply_costheta_operator, apply_sintdtheta_operator,
    laplacian_matrix, apply_laplacian,
    
    # Single-m transforms
    spat_to_sh_ml, sh_to_spat_ml, spat_to_sphtor_ml, sphtor_to_spat_ml,
    
    # Truncated transforms
    spat_to_sh_l, sh_to_spat_l, sphtor_to_spat_l, spat_to_sphtor_l,
    sh_to_grad_spat_l, spat_to_shqst_l, shqst_to_spat_l,
    
    # Profiling functions
    shtns_profiling, shtns_profiling_read_time, sh_to_spat_time, spat_to_sh_time,
    benchmark_transform, profile_memory_usage, get_profiling_summary, reset_profiling,
    compare_performance, print_profiling_report,
    
    # Robert form functions
    set_robert_form!, is_robert_form, sphtor_to_spat_robert!, spat_to_sphtor_robert!,
    robert_form_factor, apply_robert_form_to_field!

include("types.jl")
include("gauss_legendre.jl") 
include("fft_utils.jl")
include("core_transforms.jl")
include("vector_transforms.jl")
include("complex_transforms.jl")
include("utilities.jl")
include("grid_utils.jl")
include("threading.jl")
include("point_evaluation.jl")
include("special_functions.jl")
include("matrix_operators.jl")
include("single_m_transforms.jl")
include("truncated_transforms.jl")
include("profiling.jl")
include("robert_form.jl")


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
