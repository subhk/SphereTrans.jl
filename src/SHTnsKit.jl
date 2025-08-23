
module SHTnsKit

using LinearAlgebra
using FFTW
using Base.Threads

include("layout.jl")
include("gausslegendre.jl")
include("legendre.jl")
include("config.jl")
include("transform.jl")

export SHTConfig, create_gauss_config, destroy_config
export analysis, synthesis
export spat_to_SH, SH_to_spat, spat_to_SH_l, SH_to_spat_l, spat_to_SH_ml, SH_to_spat_ml, SH_to_point
export nlm_calc, nlm_cplx_calc, LM_index, LiM_index, im_from_lm

end # module SHTnsKit
