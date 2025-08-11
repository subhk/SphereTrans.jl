module SHTnsKit

export SHTnsConfig, SHTnsFlags, create_config, set_grid, sh_to_spat, spat_to_sh, free_config,
       get_lmax, get_mmax, get_nlat, get_nphi, get_nlm, lmidx,
       allocate_spectral, allocate_spatial, analyze, analyze!, synthesize, synthesize!,
       analyze_gpu, synthesize_gpu,
       enable_native_vec!, is_native_vec_enabled, synthesize_vec!, analyze_vec!,
       grid_latitudes, grid_longitudes,
       
       # Platform support functions
       check_platform_support, get_platform_description, warn_if_problematic_platform,

       # Grid and coordinates
       get_theta, get_phi, get_gauss_weights,

       # Complex field transforms
       cplx_sh_to_spat, cplx_spat_to_sh,

       # Vector transforms  
       SHsphtor_to_spat, spat_to_SHsphtor, SHsph_to_spat, SHtor_to_spat,

       # Rotations
       rotation_wigner, rotate_to_grid,

       # Multipole analysis
       multipole, power_spectrum,

       # On-the-fly transforms
       set_size,

       # OpenMP threading
       set_num_threads, get_num_threads, set_optimal_threads,

       # GPU acceleration
       gpu_init, gpu_finalize, gpu_sh_to_spat, gpu_spat_to_sh,
       initialize_gpu, cleanup_gpu,

       # High-level complex transforms
       allocate_complex_spectral, allocate_complex_spatial,
       synthesize_complex, analyze_complex,

       # High-level vector transforms
       synthesize_vector, analyze_vector, compute_gradient, compute_curl,
       # High-level rotation functions

       rotate_field, rotate_spatial_field,
       # Utility grid creation functions
       create_gauss_config, create_regular_config, create_gpu_config,
       # Helper functions for automatic differentiation
       get_lm_from_index, get_index_from_lm,
       # Library path management
       set_library_path, get_library_path, validate_library, find_system_library

    include("platform_support.jl")
    include("api.jl")
    include("highlevel.jl")
    include("utils.jl")

    # Warn about platform compatibility issues when the module is loaded
    function __init__()
        warn_if_problematic_platform()
    end

end
