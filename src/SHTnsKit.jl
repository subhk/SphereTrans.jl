module SHTnsKit

export SHTnsConfig, create_config, set_grid, sh_to_spat, spat_to_sh, free_config,
       get_lmax, get_mmax, get_nlat, get_nphi, get_nlm, lmidx

include("api.jl")

include("utils.jl")

end
