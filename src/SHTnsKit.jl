module SHTnsKit

export SHTnsConfig, create_config, set_grid, sh_to_spat, spat_to_sh, free_config,
       get_lmax, get_mmax, get_nlat, get_nphi, get_nlm, lmidx,
       allocate_spectral, allocate_spatial, analyze, analyze!, synthesize, synthesize!,
       analyze_gpu, synthesize_gpu,
       enable_native_vec!, is_native_vec_enabled, synthesize_vec!, analyze_vec!,
       grid_latitudes, grid_longitudes

include("api.jl")
include("highlevel.jl")

include("utils.jl")

end
