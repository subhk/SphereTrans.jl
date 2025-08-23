
module SHTnsKit

using LinearAlgebra
using FFTW
using Base.Threads

include("layout.jl")
include("gausslegendre.jl")
include("legendre.jl")
include("config.jl")
include("transform.jl")
include("complex_packed.jl")
include("api_compat.jl")

export SHTConfig, create_gauss_config, destroy_config
export analysis, synthesis
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm, LM_cplx_index
export spat_cplx_to_SH, SH_to_spat_cplx, SH_to_point_cplx
export shtns_verbose, shtns_print_version, shtns_get_build_info
export shtns_init, shtns_create, shtns_set_grid, shtns_set_grid_auto, shtns_create_with_grid
export shtns_use_threads, shtns_reset, shtns_destroy, shtns_unset_grid, shtns_robert_form
export sh00_1, sh10_ct, sh11_st, shlm_e1, shtns_gauss_wts

end # module SHTnsKit
