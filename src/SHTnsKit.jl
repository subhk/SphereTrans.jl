
module SHTnsKit

using LinearAlgebra
using FFTW
using Base.Threads

include("gausslegendre.jl")
include("legendre.jl")
include("config.jl")
include("transform.jl")

export SHTConfig, create_gauss_config, destroy_config
export analysis, synthesis

end # module SHTnsKit
