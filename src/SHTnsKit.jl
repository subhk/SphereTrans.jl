
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
include("vector.jl")
include("operators.jl")
include("rotations.jl")
include("local.jl")
include("api_compat.jl")

export SHTConfig, create_gauss_config, destroy_config
export analysis, synthesis
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm, LM_cplx_index
export spat_cplx_to_SH, SH_to_spat_cplx, SH_to_point_cplx
export spat_to_SHsphtor, SHsphtor_to_spat, SHsph_to_spat, SHtor_to_spat, SH_to_grad_spat
export spat_to_SHqst, SHqst_to_spat, spat_cplx_to_SHqst, SHqst_to_spat_cplx
export SHsphtor_to_spat_l, spat_to_SHsphtor_l, SHsph_to_spat_l, SHtor_to_spat_l
export spat_to_SHsphtor_ml, SHsphtor_to_spat_ml
export spat_to_SHqst_l, SHqst_to_spat_l, spat_to_SHqst_ml, SHqst_to_spat_ml
export mul_ct_matrix, st_dt_matrix, SH_mul_mx
export SH_to_lat, SHqst_to_lat
export SH_Zrotate
export shtns_verbose, shtns_print_version, shtns_get_build_info
export shtns_init, shtns_create, shtns_set_grid, shtns_set_grid_auto, shtns_create_with_grid
export shtns_use_threads, shtns_reset, shtns_destroy, shtns_unset_grid, shtns_robert_form
export sh00_1, sh10_ct, sh11_st, shlm_e1, shtns_gauss_wts

end # module SHTnsKit
